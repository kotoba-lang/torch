(ns torch.continuous-test
  (:require [clojure.test :refer [deftest is]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [num.tensor :as t]
            [torch.continuous :as continuous]
            [torch.kv-cache :as kv]
            [torch.model :as model]
            [torch.num-backend :as nb]
            [torch.paged-runtime :as paged]))

(def backend (cpu/cpu-backend))

(defn- cpu-storage [block-count block-size heads kv-heads]
  (let [keys (atom (vec (repeat block-count (vec (repeat block-size nil)))))
        values (atom (vec (repeat block-count (vec (repeat block-size nil)))))]
    {:write! (fn [key value block offset]
               (swap! keys assoc-in [block offset] (arr/->vec key))
               (swap! values assoc-in [block offset] (arr/->vec value)))
     :copy-block! (fn [source destination tokens]
                    (doseq [offset (range tokens)]
                      (swap! keys assoc-in [destination offset]
                             (get-in @keys [source offset]))
                      (swap! values assoc-in [destination offset]
                             (get-in @values [source offset]))))
     :attention
     (fn [query blocks length]
       (let [addresses (for [position (range length)]
                         [(nth blocks (quot position block-size))
                          (mod position block-size)])
             key-data (vec (mapcat #(get-in @keys %) addresses))
             value-data (vec (mapcat #(get-in @values %) addresses))
             kv-width (quot (count key-data) length)]
         (t/multi-head-attention
          query (arr/from-vec backend key-data [length kv-width])
          (arr/from-vec backend value-data [length kv-width])
          heads {:kv-heads kv-heads})))}))

(deftest ragged-continuous-paged-llama-admits-releases-and-reuses
  (let [model* (model/sequential
                (model/embedding 6 4)
                (model/llama-block 4 2 8 {:kv-heads 1})
                (model/llama-block 4 2 8 {:kv-heads 1})
                (model/rmsnorm 4) (model/lm-head 4 6))
        weights (nb/random-weights backend model* 89)
        runtimes (mapv (fn [_]
                         (paged/runtime (kv/pool 3 2)
                                        (cpu-storage 3 2 2 1)))
                       (range 2))
        calls (atom [])
        step-fn (fn [token runtimes request-id]
                  (swap! calls conj [request-id token])
                  (let [step (nb/llama-lm-paged-step
                              model* weights (arr/from-vec backend [token] [1])
                              runtimes request-id)]
                    (assoc step :logits (arr/->vec (:logits step)))))
        admitted (-> (continuous/engine runtimes step-fn 2)
                     (continuous/enqueue :a [2]
                                         {:temperature 0.0 :max-new-tokens 1
                                          :eos-id -1})
                     (continuous/enqueue :b [1 3 2]
                                         {:temperature 0.0 :max-new-tokens 2
                                          :eos-id -1})
                     (continuous/enqueue :c [4]
                                         {:temperature 0.0 :max-new-tokens 1
                                          :eos-id -1})
                     continuous/admit)]
    (is (= [:a :b] (:order admitted)))
    (is (= [:c] (mapv :id (:waiting admitted))))
    (is (every? empty? (map #(get-in % [:pool :free]) (:runtimes admitted))))
    (let [tick1 (continuous/tick admitted)]
      (is (= [:b :c] (:order tick1))
          "a releases its block and c is immediately ragged-prefilled")
      (is (= :length (get-in tick1 [:completed :a :reason])))
      (is (= 4 (get-in tick1 [:runtimes 0 :pool :sequences :b :length])))
      (is (= 1 (get-in tick1 [:runtimes 0 :pool :sequences :c :length])))
      (let [tick2 (continuous/tick tick1)]
        (is (empty? (:running tick2)))
        (is (= #{:a :b :c} (set (keys (:completed tick2))))
            "different prompt lengths and completion times share one pool")
        (is (every? #(= #{0 1 2} (set (get-in % [:pool :free])))
                    (:runtimes tick2)))
        (is (every? #(kv/valid? (:pool %)) (:runtimes tick2)))
        (is (= 6 (count @calls)))))))

(deftest head-of-line-request-waits-without-mutating-layer-pools
  (let [storage {:write! (fn [& _]) :copy-block! (fn [& _])
                 :attention (fn [& _] [0.0])}
        runtime (paged/runtime (kv/pool 1 2) storage)
        engine (-> (continuous/engine [runtime]
                                      (fn [_ runtimes _]
                                        {:logits [1.0] :runtimes runtimes}) 1)
                   (continuous/enqueue :large [1 2 3])
                   (continuous/enqueue :small [1])
                   continuous/admit)]
    (is (empty? (:running engine)))
    (is (= [:large :small] (mapv :id (:waiting engine))))
    (is (= #{0} (set (get-in engine [:runtimes 0 :pool :free]))))
    (is (kv/valid? (get-in engine [:runtimes 0 :pool])))))
