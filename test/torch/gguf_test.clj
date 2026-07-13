(ns torch.gguf-test
  (:require [clojure.test :refer [deftest is testing]]
            [num.array :as arr]
            [num.cpu :as cpu]
            [num.dtype :as dtype]
            [torch.core :as core]
            [torch.gguf :as gguf]
            [torch.model :as model]
            [torch.num-backend :as nb]
            [torch.tokenizer :as tokenizer])
  (:import [java.io ByteArrayOutputStream]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.charset StandardCharsets]
           [java.nio.file Files]))

(defn- le-bytes [n put]
  (let [buffer (doto (ByteBuffer/allocate n) (.order ByteOrder/LITTLE_ENDIAN))]
    (put buffer)
    (.array buffer)))

(defn- write! [^ByteArrayOutputStream out bytes] (.write out ^bytes bytes))
(defn- u32! [out value] (write! out (le-bytes 4 #(.putInt % (int value)))))
(defn- u64! [out value] (write! out (le-bytes 8 #(.putLong % (long value)))))
(defn- f32! [out value] (write! out (le-bytes 4 #(.putFloat % (float value)))))
(defn- f16! [out value]
  (write! out (le-bytes 2 #(.putShort % (short (dtype/f32->f16-bits value))))))
(defn- string! [out value]
  (let [bytes (.getBytes ^String value StandardCharsets/UTF_8)]
    (u64! out (alength bytes)) (write! out bytes)))
(defn- pad-to! [^ByteArrayOutputStream out alignment]
  (dotimes [_ (mod (- alignment (mod (.size out) alignment)) alignment)] (.write out 0)))

(defn- fixture []
  (let [out (ByteArrayOutputStream.)]
    (write! out (.getBytes "GGUF" StandardCharsets/US_ASCII))
    (u32! out 3) (u64! out 3) (u64! out 3)
    ;; metadata
    (string! out "general.alignment") (u32! out 4) (u32! out 32)
    (string! out "llama.block_count") (u32! out 4) (u32! out 2)
    (string! out "tokenizer.ggml.tokens") (u32! out 9)
    (u32! out 8) (u64! out 2) (string! out "a") (string! out "猫")
    ;; tensor directory: GGML dimensions are fastest-axis first
    (string! out "f32.weight") (u32! out 2) (u64! out 2) (u64! out 2)
    (u32! out 0) (u64! out 0)
    (string! out "f16.weight") (u32! out 1) (u64! out 4)
    (u32! out 1) (u64! out 32)
    (string! out "q8.weight") (u32! out 1) (u64! out 32)
    (u32! out 8) (u64! out 64)
    (pad-to! out 32)
    ;; tensor data at relative offsets 0, 32, 64
    (doseq [v [1.0 -2.0 3.5 0.25]] (f32! out v))
    (pad-to! out 32)
    (doseq [v [0.5 -1.0 2.0 4.0]] (f16! out v))
    (pad-to! out 32)
    (f16! out 0.25)
    (doseq [v (range -16 16)] (.write out (bit-and v 0xff)))
    (.toByteArray out)))

(defn- q4-k-fixture []
  (let [out (ByteArrayOutputStream.)]
    (write! out (.getBytes "GGUF" StandardCharsets/US_ASCII))
    (u32! out 3) (u64! out 1) (u64! out 1)
    (string! out "general.alignment") (u32! out 4) (u32! out 32)
    (string! out "q4-k.weight") (u32! out 1) (u64! out 256)
    (u32! out 12) (u64! out 0)
    (pad-to! out 32)
    ;; d=.5, dmin=.25. All eight 6-bit scale/min pairs exercise both packing
    ;; layouts, including upper bits stored in the first eight bytes.
    (f16! out 0.5) (f16! out 0.25)
    ;; scales [1 2 3 4 49 34 19 60], mins [9 10 11 12 45 30 51 20]
    (doseq [v [193 130 67 196, 137 74 203 76, 209 226 51 76]] (.write out v))
    ;; Each byte holds one low-nibble value followed 32 positions later by its
    ;; high-nibble value. 0x21 therefore yields q=1 then q=2.
    (dotimes [_ 128] (.write out 0x21))
    (.toByteArray out)))

(defn- q6-k-fixture []
  (let [out (ByteArrayOutputStream.)]
    (write! out (.getBytes "GGUF" StandardCharsets/US_ASCII))
    (u32! out 3) (u64! out 1) (u64! out 1)
    (string! out "general.alignment") (u32! out 4) (u32! out 32)
    (string! out "q6-k.weight") (u32! out 1) (u64! out 256)
    (u32! out 14) (u64! out 0)
    (pad-to! out 32)
    ;; Zero low/high quant bits decode to q=-32. Signed scales -8..7 make
    ;; every 16-value subblock independently observable with d=.25.
    (dotimes [_ (+ 128 64)] (.write out 0))
    (doseq [scale (range -8 8)] (.write out (bit-and scale 0xff)))
    (f16! out 0.25)
    (.toByteArray out)))

(deftest parses-v3-metadata-directory-and-alignment
  (let [file (gguf/parse-bytes (fixture))]
    (is (= 3 (:version file)))
    (is (= 32 (:alignment file)))
    (is (= 2 (get-in file [:metadata "llama.block_count"])))
    (is (= ["a" "猫"] (get-in file [:metadata "tokenizer.ggml.tokens"])))
    (is (= [2 2] (get-in file [:tensor-map "f32.weight" :shape])))
    (is (zero? (mod (:data-offset file) 32)))))

(deftest decodes-f32-f16-and-q8-zero
  (let [file (gguf/parse-bytes (fixture))]
    (is (= [1.0 -2.0 3.5 0.25]
           (:values (gguf/read-tensor file "f32.weight"))))
    (is (= [0.5 -1.0 2.0 4.0]
           (:values (gguf/read-tensor file "f16.weight"))))
    (is (= (mapv #(* 0.25 %) (range -16 16))
           (:values (gguf/read-tensor file "q8.weight"))))))

(deftest decodes-q4-k-packed-scales-mins-and-nibbles
  (let [tensor (gguf/read-tensor (gguf/parse-bytes (q4-k-fixture))
                                 "q4-k.weight")
        values (:values tensor)]
    (is (= :q4-k (:type tensor)))
    (is (= [256] (:shape tensor)))
    (is (= (vec (mapcat (fn [index]
                          (let [scales [1 2 3 4 49 34 19 60]
                                mins [9 10 11 12 45 30 51 20]
                                quant (if (even? index) 1 2)]
                            (repeat 32 (- (* 0.5 (nth scales index) quant)
                                          (* 0.25 (nth mins index))))))
                        (range 8)))
           values))))

(deftest quantized-tensors-require-complete-blocks-per-row
  (let [file (gguf/parse-bytes (q4-k-fixture))
        malformed (assoc-in file [:tensor-map "q4-k.weight" :shape] [2 128])]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"row width must be divisible"
                          (gguf/read-tensor malformed "q4-k.weight")))))

(deftest decodes-q6-k-low-high-bits-and-signed-scales
  (let [tensor (gguf/read-tensor (gguf/parse-bytes (q6-k-fixture))
                                 "q6-k.weight")]
    (is (= :q6-k (:type tensor)))
    (is (= (vec (mapcat #(repeat 16 (* -8.0 %)) (range -8 8)))
           (:values tensor)))))

(deftest file-backed-loader-uses-positional-ranges
  (let [path (Files/createTempFile "torch-gguf-" ".gguf"
                                   (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (Files/write path (fixture) (make-array java.nio.file.OpenOption 0))
      (let [file (gguf/load-file path)]
        (is (= 3 (:version file)))
        (is (= [0.5 -1.0 2.0 4.0]
               (:values (gguf/read-tensor file "f16.weight")))))
      (finally (Files/deleteIfExists path)))))

(deftest rejects-corrupt-and-unsupported-input
  (testing "magic and tensor type failures include structured context"
    (is (thrown? Exception (gguf/parse-bytes (.getBytes "nope"))))
    (let [file (gguf/parse-bytes (fixture))
          file (assoc-in file [:tensor-map "q8.weight" :type] :unsupported)]
      (is (thrown? Exception (gguf/read-tensor file "q8.weight")))
      (is (thrown? Exception (gguf/read-tensor file "missing"))))))

(deftest constructs-llama-model-and-tokenizer-from-metadata
  (let [file {:metadata
              {"general.architecture" "llama"
               "llama.block_count" 2
               "llama.embedding_length" 4
               "llama.feed_forward_length" 8
               "llama.attention.head_count" 2
               "llama.attention.head_count_kv" 1
               "llama.context_length" 128
               "llama.rope.freq_base" 500000.0
               "tokenizer.ggml.tokens" ["<unk>" "<bos>" "<eos>" "a" "b" "ab"]
               "tokenizer.ggml.merges" ["a b"]
               "tokenizer.ggml.unknown_token_id" 0
               "tokenizer.ggml.bos_token_id" 1
               "tokenizer.ggml.eos_token_id" 2}}
        model* (gguf/llama-model file)
        layers (model/execution-layers model*)
        tokenizer* (gguf/gguf-tokenizer file)]
    (is (= [:embedding :llama-block :llama-block :rmsnorm :lm-head]
           (mapv model/layer-type layers)))
    (is (= 500000.0 (get-in (second layers) [:llama-block 3 :rope-theta])))
    (is (= 1 (get-in (second layers) [:llama-block 3 :kv-heads])))
    (is (= [1 5] (tokenizer/encode tokenizer* "ab")))
    (is (= "ab" (tokenizer/decode tokenizer* [1 5 2])))))

(deftest maps-standard-llama-tensor-names-and-orientations
  (let [metadata {"general.architecture" "llama"
                  "llama.block_count" 1 "llama.embedding_length" 4
                  "llama.feed_forward_length" 8 "llama.attention.head_count" 2
                  "llama.attention.head_count_kv" 1
                  "llama.context_length" 32
                  "tokenizer.ggml.tokens" ["0" "1" "2" "3" "4" "5"]}
        shapes {"token_embd.weight" [6 4]
                "blk.0.attn_norm.weight" [4]
                "blk.0.attn_q.weight" [4 4] "blk.0.attn_k.weight" [2 4]
                "blk.0.attn_v.weight" [2 4] "blk.0.attn_output.weight" [4 4]
                "blk.0.ffn_norm.weight" [4]
                "blk.0.ffn_gate.weight" [8 4] "blk.0.ffn_up.weight" [8 4]
                "blk.0.ffn_down.weight" [4 8]
                "output_norm.weight" [4]}
        fake-read (fn [_ name]
                    (let [shape (get shapes name) n (reduce * shape)]
                      {:shape shape :values
                       (if (= 1 (count shape)) (vec (repeat n 1.0))
                           (mapv #(* 0.01 (inc %)) (range n)))}))
        backend (cpu/cpu-backend)
        file {:metadata metadata :tensor-map (into {} (map #(vector % {}) (keys shapes)))}]
    (with-redefs [gguf/read-tensor fake-read]
      (let [model* (gguf/llama-model file)
            weights (gguf/load-llama-weights file backend)
            output (core/run (nb/num-backend backend weights) model*
                             (arr/from-vec backend [2 0] [2]))]
        (is (= 4 (count weights)))
        (is (= [4 2] (:shape (:kw (second weights)))))
        (is (= [4 2] (:shape (:vw (second weights)))))
        (is (= [4 8] (:shape (:gate (second weights)))))
        (is (= [8 4] (:shape (:down (second weights)))))
        (is (= [4 6] (:shape (:w (last weights)))))
        (is (= [2 6] (:shape output)))))))
