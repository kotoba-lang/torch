(ns torch.safetensors-test
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [torch.core :as core]
            [torch.model :as model]
            [torch.num-backend :as nb]
            [torch.safetensors :as safe]
            [torch.state-dict :as state])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.charset StandardCharsets]
           [java.nio.file Files OpenOption]))

(def backend (cpu/cpu-backend))

(defn- f32-bytes [values]
  (let [buffer (doto (ByteBuffer/allocate (* 4 (count values)))
                 (.order ByteOrder/LITTLE_ENDIAN))]
    (doseq [value values] (.putFloat buffer (float value)))
    (.array buffer)))

(defn- write-f32-checkpoint [tensors]
  (let [[header payload _]
        (reduce (fn [[header chunks offset] [name array]]
                  (let [values (arr/->vec array) bytes (f32-bytes values)
                        end (+ offset (alength bytes))]
                    [(assoc header name {"dtype" "F32" "shape" (:shape array)
                                         "data_offsets" [offset end]})
                     (conj chunks bytes) end]))
                [{"__metadata__" {"format" "pt"}} [] 0]
                (sort-by key tensors))
        header-bytes (.getBytes (json/write-str header) StandardCharsets/UTF_8)
        total (+ 8 (alength header-bytes) (reduce + (map alength payload)))
        buffer (doto (ByteBuffer/allocate total) (.order ByteOrder/LITTLE_ENDIAN))
        path (Files/createTempFile "torch-state-" ".safetensors"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (.putLong buffer (alength header-bytes))
    (.put buffer header-bytes)
    (doseq [bytes payload] (.put buffer bytes))
    (Files/write path (.array buffer) (make-array OpenOption 0))
    path))

(defn- write-mixed-checkpoint []
  (let [header {"f32" {"dtype" "F32" "shape" [1] "data_offsets" [0 4]}
                "f16" {"dtype" "F16" "shape" [1] "data_offsets" [4 6]}
                "bf16" {"dtype" "BF16" "shape" [1] "data_offsets" [6 8]}}
        header-bytes (.getBytes (json/write-str header) StandardCharsets/UTF_8)
        buffer (doto (ByteBuffer/allocate (+ 8 (alength header-bytes) 8))
                 (.order ByteOrder/LITTLE_ENDIAN))
        path (Files/createTempFile "torch-mixed-" ".safetensors"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (.putLong buffer (alength header-bytes))
    (.put buffer header-bytes)
    (.putFloat buffer (float 1.5))
    (.putShort buffer (short 0x3e00))
    (.putShort buffer (short 0x3fc0))
    (Files/write path (.array buffer) (make-array OpenOption 0))
    path))

(deftest loads-real-safetensors-file-and-preserves-inference
  (let [model* (model/sequential (model/linear 4 4)
                                 (model/multihead-attention 4 2))
        weights (nb/random-weights backend model* 53)
        checkpoint (write-f32-checkpoint (state/state-dict model* weights))
        input (arr/from-vec backend
                            [0.2 -0.1 0.3 0.4, -0.2 0.1 0.5 -0.3] [2 4])]
    (try
      (let [loaded (safe/load-weights checkpoint backend model*)
            expected (core/run (nb/num-backend backend weights) model* input)
            actual (core/run (nb/num-backend backend loaded) model* input)]
        (is (= [2 4] (:shape actual)))
        (is (every? #(< (Math/abs %) 1.0e-6)
                    (map - (arr/->vec expected) (arr/->vec actual)))))
      (finally (Files/deleteIfExists checkpoint)))))

(deftest strict-loader-rejects-unexpected-tensors
  (let [model* (model/sequential (model/linear 2 2))
        weights (nb/random-weights backend model* 5)
        external (state/state-dict model* weights)
        extra (assoc external "unused.weight" (arr/from-vec backend [1.0] [1]))
        checkpoint (write-f32-checkpoint extra)]
    (try
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unexpected tensors"
                            (safe/load-weights checkpoint backend model*)))
      (is (= 1 (count (safe/load-weights checkpoint backend model*
                                      {:strict? false}))))
      (finally (Files/deleteIfExists checkpoint)))))

(deftest decodes-f32-f16-and-bf16-windows
  (let [checkpoint-path (write-mixed-checkpoint)]
    (try
      (with-open [checkpoint (safe/open-file checkpoint-path)]
        (is (= ["bf16" "f16" "f32"] (safe/tensor-names checkpoint)))
        (doseq [name ["f32" "f16" "bf16"]]
          (is (< (Math/abs (- 1.5
                             (first (arr/->vec
                                     (safe/read-tensor checkpoint backend name)))))
                 1.0e-6)
              name)))
      (finally (Files/deleteIfExists checkpoint-path)))))
