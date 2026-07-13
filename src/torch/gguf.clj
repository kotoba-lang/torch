(ns torch.gguf
  "GGUF v2/v3 metadata and tensor loader with long file offsets."
  (:refer-clojure :exclude [load-file])
  (:require [clojure.string :as str]
            [num.array :as arr]
            [num.dtype :as dtype]
            [torch.model :as model]
            [torch.tokenizer :as tokenizer])
  (:import [java.io RandomAccessFile]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.charset StandardCharsets]
           [java.nio.file Files Path Paths]))

(def value-types
  {0 :uint8 1 :int8 2 :uint16 3 :int16 4 :uint32 5 :int32
   6 :float32 7 :bool 8 :string 9 :array 10 :uint64 11 :int64 12 :float64})

(def tensor-types {0 :f32 1 :f16 8 :q8-0})

(defn- source-size [{:keys [bytes path]}]
  (if bytes (alength ^bytes bytes) (Files/size ^Path path)))

(defn- read-range [{:keys [bytes path] :as source} offset n]
  (let [size (source-size source)]
    (when-not (and (<= 0 offset) (<= 0 n) (<= (+ offset n) size)
                   (<= n Integer/MAX_VALUE))
      (throw (ex-info "GGUF read is outside source bounds"
                      {:offset offset :count n :size size})))
    (if bytes
      (java.util.Arrays/copyOfRange ^bytes bytes (int offset) (int (+ offset n)))
      (with-open [file (RandomAccessFile. (.toFile ^Path path) "r")]
        (.seek file offset)
        (let [output (byte-array (int n))]
          (.readFully file output)
          output)))))

(defn- cursor [source] {:source source :position (atom 0)})

(defn- take-bytes [{:keys [source position]} n]
  (let [offset @position bytes (read-range source offset n)]
    (swap! position + n)
    bytes))

(defn- little-buffer [bytes]
  (doto (ByteBuffer/wrap bytes) (.order ByteOrder/LITTLE_ENDIAN)))

(defn- u8 [reader] (bit-and 0xff (aget ^bytes (take-bytes reader 1) 0)))
(defn- i8 [reader] (int (aget ^bytes (take-bytes reader 1) 0)))
(defn- u16 [reader] (bit-and 0xffff (.getShort (little-buffer (take-bytes reader 2)))))
(defn- i16 [reader] (.getShort (little-buffer (take-bytes reader 2))))
(defn- u32 [reader] (bit-and 0xffffffff (.getInt (little-buffer (take-bytes reader 4)))))
(defn- i32 [reader] (.getInt (little-buffer (take-bytes reader 4))))
(defn- u64 [reader]
  (let [value (.getLong (little-buffer (take-bytes reader 8)))]
    (when (neg? value)
      (throw (ex-info "GGUF uint64 exceeds signed JVM range" {:value value})))
    value))
(defn- i64 [reader] (.getLong (little-buffer (take-bytes reader 8))))
(defn- f32 [reader] (.getFloat (little-buffer (take-bytes reader 4))))
(defn- f64 [reader] (.getDouble (little-buffer (take-bytes reader 8))))

(defn- gguf-string [reader]
  (let [n (u64 reader)]
    (when (> n Integer/MAX_VALUE)
      (throw (ex-info "GGUF string is too large" {:bytes n})))
    (String. ^bytes (take-bytes reader n) StandardCharsets/UTF_8)))

(declare read-value)

(defn- read-value [reader type-id]
  (case (get value-types type-id)
    :uint8 (u8 reader) :int8 (i8 reader)
    :uint16 (u16 reader) :int16 (i16 reader)
    :uint32 (u32 reader) :int32 (i32 reader)
    :float32 (f32 reader)
    :bool (let [value (u8 reader)]
            (when-not (#{0 1} value)
              (throw (ex-info "GGUF boolean must be encoded as 0 or 1"
                              {:value value})))
            (= value 1))
    :string (gguf-string reader) :uint64 (u64 reader) :int64 (i64 reader)
    :float64 (f64 reader)
    :array (let [element-type (u32 reader) n (u64 reader)]
             (when (> n Integer/MAX_VALUE)
               (throw (ex-info "GGUF metadata array is too large" {:count n})))
             (mapv (fn [_] (read-value reader element-type)) (range n)))
    (throw (ex-info "unknown GGUF metadata value type" {:type-id type-id}))))

(defn- align-up [value alignment]
  (* (quot (+ value (dec alignment)) alignment) alignment))

(defn- parse-source [source]
  (let [reader (cursor source)
        magic (String. ^bytes (take-bytes reader 4) StandardCharsets/US_ASCII)
        _ (when-not (= magic "GGUF")
            (throw (ex-info "not a GGUF file" {:magic magic})))
        version (u32 reader)
        _ (when-not (#{2 3} version)
            (throw (ex-info "unsupported GGUF version" {:version version})))
        tensor-count (u64 reader) metadata-count (u64 reader)
        metadata (into {}
                       (repeatedly metadata-count
                                   (fn [] (let [key (gguf-string reader)
                                                type-id (u32 reader)]
                                            [key (read-value reader type-id)]))))
        tensors (mapv (fn [_]
                        (let [name (gguf-string reader)
                              rank (u32 reader)
                              dimensions (mapv (fn [_] (u64 reader)) (range rank))
                              type-id (u32 reader)
                              offset (u64 reader)]
                          {:name name :rank rank :ggml-shape dimensions
                           :shape (vec (reverse dimensions))
                           :type-id type-id :type (get tensor-types type-id :unsupported)
                           :offset offset}))
                      (range tensor-count))
        alignment (long (get metadata "general.alignment" 32))
        _ (when-not (and (pos? alignment) (zero? (bit-and alignment (dec alignment))))
            (throw (ex-info "GGUF alignment must be a positive power of two"
                            {:alignment alignment})))
        data-offset (align-up @(:position reader) alignment)
        _ (when-not (= (count metadata) metadata-count)
            (throw (ex-info "GGUF contains duplicate metadata keys" {})))
        _ (when-not (every? #(zero? (mod (:offset %) alignment)) tensors)
            (throw (ex-info "GGUF tensor offset violates general.alignment"
                            {:alignment alignment})))
        tensors (mapv #(assoc % :data-offset (+ data-offset (:offset %))) tensors)]
    (when-not (= (count (set (map :name tensors))) tensor-count)
      (throw (ex-info "GGUF contains duplicate tensor names" {})))
    {:version version :metadata metadata :tensors tensors
     :tensor-map (into {} (map (juxt :name identity) tensors))
     :alignment alignment :data-offset data-offset :source source}))

(defn parse-bytes [bytes]
  (parse-source {:bytes bytes}))

(defn load-file [path]
  (parse-source {:path (Paths/get (str path) (make-array String 0))}))

(defn- nelems [shape] (reduce * 1 shape))

(defn- tensor-byte-count [{:keys [type shape]}]
  (let [n (nelems shape)]
    (case type
      :f32 (* 4 n)
      :f16 (* 2 n)
      :q8-0 (do (when-not (zero? (mod n 32))
                  (throw (ex-info "Q8_0 tensor element count must divide 32" {:count n})))
                (* 34 (quot n 32)))
      (throw (ex-info "unsupported GGUF tensor type" {:type type})))))

(defn- decode-tensor [{:keys [type shape data-offset] :as tensor} source]
  (let [bytes (read-range source data-offset (tensor-byte-count tensor))
        buffer (little-buffer bytes)
        n (nelems shape)]
    (case type
      :f32 (mapv (fn [_] (double (.getFloat buffer))) (range n))
      :f16 (mapv (fn [_] (dtype/f16-bits->f32 (.getShort buffer))) (range n))
      :q8-0
      (vec (mapcat (fn [_]
                     (let [scale (dtype/f16-bits->f32 (.getShort buffer))]
                       (mapv (fn [_] (* scale (int (.get buffer)))) (range 32))))
                   (range (quot n 32)))))))

(defn read-tensor
  "Decode a named F32/F16/Q8_0 tensor to `{:shape :values :type}`."
  [gguf name]
  (let [tensor (get-in gguf [:tensor-map name])]
    (when-not tensor
      (throw (ex-info "GGUF tensor not found" {:name name})))
    (assoc tensor :values (decode-tensor tensor (:source gguf)))))

(defn llama-config
  "Extract the Llama architecture fields required by torch's decoder model."
  [gguf]
  (let [m (:metadata gguf)
        architecture (get m "general.architecture")
        head-count (get m "llama.attention.head_count")
        config {:architecture architecture
                :block-count (get m "llama.block_count")
                :embed-dim (get m "llama.embedding_length")
                :hidden-dim (get m "llama.feed_forward_length")
                :head-count head-count
                :kv-head-count (get m "llama.attention.head_count_kv" head-count)
                :context-length (get m "llama.context_length")
                :rope-theta (double (get m "llama.rope.freq_base" 10000.0))}]
    (when-not (and (= architecture "llama")
                   (every? pos-int? ((juxt :block-count :embed-dim :hidden-dim
                                           :head-count :kv-head-count
                                           :context-length) config))
                   (zero? (mod (:embed-dim config) (:head-count config)))
                   (zero? (mod (:head-count config) (:kv-head-count config))))
      (throw (ex-info "GGUF lacks a valid Llama architecture configuration"
                      config)))
    config))

(defn llama-model
  "Construct a complete torch.model Llama LM description from GGUF metadata."
  [gguf]
  (let [{:keys [block-count embed-dim hidden-dim head-count kv-head-count rope-theta]}
        (llama-config gguf)
        vocab (count (get-in gguf [:metadata "tokenizer.ggml.tokens"]))]
    (apply model/sequential
           (concat [(model/embedding vocab embed-dim)]
                   (repeat block-count
                           (model/llama-block embed-dim head-count hidden-dim
                                              {:rope-theta rope-theta
                                               :kv-heads kv-head-count}))
                   [(model/rmsnorm embed-dim) (model/lm-head embed-dim vocab)]))))

(defn gguf-tokenizer
  "Build torch.tokenizer from standard GGUF tokenizer metadata."
  [gguf]
  (let [m (:metadata gguf)
        tokens (vec (get m "tokenizer.ggml.tokens"))
        merges (mapv #(vec (str/split % #" " 2))
                     (get m "tokenizer.ggml.merges" []))]
    (tokenizer/tokenizer
     {:tokens tokens :merges merges
      :unk-id (get m "tokenizer.ggml.unknown_token_id")
      :bos-id (get m "tokenizer.ggml.bos_token_id")
      :eos-id (get m "tokenizer.ggml.eos_token_id")
      :add-bos? (get m "tokenizer.ggml.add_bos_token" true)
      :add-eos? (get m "tokenizer.ggml.add_eos_token" false)})))

(defn- upload-vector [gguf backend name]
  (let [{:keys [shape values]} (read-tensor gguf name)]
    (arr/from-vec backend values shape)))

(defn- upload-matrix
  "GGUF linear matrices are `[out in]`; torch.num-backend uses `[in out]`."
  [gguf backend name]
  (let [{[out in] :shape values :values} (read-tensor gguf name)
        transposed (mapv (fn [i]
                           (let [input-index (quot i out) output-index (mod i out)]
                             (nth values (+ (* output-index in) input-index))))
                         (range (* in out)))]
    (arr/from-vec backend transposed [in out])))

(defn load-llama-weights
  "Decode and upload standard Llama GGUF tensors into the weights vector used
  by `torch.num-backend`. Quantized tensors are currently dequantized to f32."
  [gguf backend]
  (let [{:keys [block-count]} (llama-config gguf)
        embedding (upload-vector gguf backend "token_embd.weight")
        blocks
        (mapv (fn [index]
                (let [prefix (str "blk." index ".")]
                  {:attn-norm (upload-vector gguf backend (str prefix "attn_norm.weight"))
                   :qw (upload-matrix gguf backend (str prefix "attn_q.weight"))
                   :kw (upload-matrix gguf backend (str prefix "attn_k.weight"))
                   :vw (upload-matrix gguf backend (str prefix "attn_v.weight"))
                   :ow (upload-matrix gguf backend (str prefix "attn_output.weight"))
                   :ffn-norm (upload-vector gguf backend (str prefix "ffn_norm.weight"))
                   :gate (upload-matrix gguf backend (str prefix "ffn_gate.weight"))
                   :up (upload-matrix gguf backend (str prefix "ffn_up.weight"))
                   :down (upload-matrix gguf backend (str prefix "ffn_down.weight"))}))
              (range block-count))
        final-norm (upload-vector gguf backend "output_norm.weight")
        output (if (contains? (:tensor-map gguf) "output.weight")
                 (upload-matrix gguf backend "output.weight")
                 ;; Tied embeddings need a transposed materialization for the
                 ;; engine's `[embed vocab]` LM-head layout.
                 (let [[vocab embed] (:shape embedding)
                       values (vec (arr/->vec embedding))
                       transposed (mapv (fn [i]
                                          (let [e (quot i vocab) v (mod i vocab)]
                                            (nth values (+ (* v embed) e))))
                                        (range (* vocab embed)))]
                   (arr/from-vec backend transposed [embed vocab])))]
    (vec (concat [{:w embedding}] blocks [{:w final-norm} {:w output}]))))
