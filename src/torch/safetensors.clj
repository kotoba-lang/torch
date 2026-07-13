(ns torch.safetensors
  "Validated safetensors reader/writer and torch state-dict loader."
  (:require [clojure.data.json :as json]
            [clojure.set :as set]
            [num.array :as arr]
            [torch.state-dict :as state])
  (:import [java.io Closeable RandomAccessFile]
           [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel]
           [java.nio.charset StandardCharsets]
           [java.nio.file Files Path StandardCopyOption StandardOpenOption]
           [java.nio.file.attribute FileAttribute]))

(def ^:private max-header-bytes (* 128 1024 1024))
(def ^:private dtype-bytes
  {"F16" 2 "BF16" 2 "F32" 4 "F64" 8
   "I8" 1 "U8" 1 "I16" 2 "U16" 2 "I32" 4 "U32" 4 "I64" 8 "U64" 8})

(defrecord SafeTensorFile
  [^RandomAccessFile file ^FileChannel channel path data-start tensors metadata]
  Closeable
  (close [_] (.close channel) (.close file)))

(defn- read-fully-at! [^FileChannel channel ^ByteBuffer buffer position]
  (loop [position (long position)]
    (when (.hasRemaining buffer)
      (let [read-count (.read channel buffer position)]
        (when (neg? read-count)
          (throw (ex-info "torch.safetensors: truncated file"
                          {:position position})))
        (recur (+ position read-count)))))
  (.flip buffer)
  buffer)

(defn- nelems [shape] (reduce * 1 (map long shape)))

(defn open-file [path]
  (let [file (RandomAccessFile. (str path) "r") channel (.getChannel file)]
    (try
      (let [length-buffer (doto (ByteBuffer/allocate 8)
                            (.order ByteOrder/LITTLE_ENDIAN))
            _ (read-fully-at! channel length-buffer 0)
            header-length (.getLong length-buffer)
            file-length (.size channel)]
        (when (or (neg? header-length) (> header-length max-header-bytes)
                  (> (+ 8 header-length) file-length))
          (throw (ex-info "torch.safetensors: invalid header length"
                          {:header-length header-length :file-length file-length})))
        (let [header-buffer (ByteBuffer/allocate (int header-length))
              _ (read-fully-at! channel header-buffer 8)
              header (json/read-str
                      (String. (.array header-buffer) StandardCharsets/UTF_8))
              metadata (get header "__metadata__" {})
              tensors (dissoc header "__metadata__")
              data-start (+ 8 header-length)
              payload-length (- file-length data-start)]
          (doseq [[name {:strs [dtype shape data_offsets] :as entry}] tensors]
            (let [[start end] data_offsets width (get dtype-bytes dtype)
                  expected (when width (* width (nelems shape)))]
              (when-not (and width (vector? shape) (= 2 (count data_offsets))
                             (integer? start) (integer? end)
                             (<= 0 start end payload-length)
                             (= expected (- end start)))
                (throw (ex-info "torch.safetensors: invalid tensor entry"
                                {:tensor name :entry entry
                                 :payload-length payload-length
                                 :expected-bytes expected})))))
          (->SafeTensorFile file channel (str path) data-start tensors metadata)))
      (catch Throwable error
        (.close channel) (.close file) (throw error)))))

(defn tensor-names [checkpoint] (vec (sort (keys (:tensors checkpoint)))))
(defn tensor-info [checkpoint name] (get (:tensors checkpoint) name))

(defn- write-fully! [^FileChannel channel ^ByteBuffer buffer]
  (while (.hasRemaining buffer) (.write channel buffer)))

(defn- padded-header-bytes [header]
  (let [raw (.getBytes (json/write-str header) StandardCharsets/UTF_8)
        padded-length (* 8 (quot (+ (alength raw) 7) 8))
        padded (byte-array padded-length)]
    (java.util.Arrays/fill padded (byte 0x20))
    (System/arraycopy raw 0 padded 0 (alength raw))
    padded))

(defn- encode-tensor [array storage-dtype]
  (let [values (arr/->vec array)
        [dtype width put-value]
        (case (or storage-dtype (:dtype array))
          :f32 ["F32" 4 #(.putFloat ^ByteBuffer %1 (float %2))]
          :f64 ["F64" 8 #(.putDouble ^ByteBuffer %1 (double %2))]
          (throw (ex-info "torch.safetensors: unsupported output dtype"
                          {:dtype (:dtype array) :supported [:f32 :f64]})))
        buffer (doto (ByteBuffer/allocate (* width (count values)))
                 (.order ByteOrder/LITTLE_ENDIAN))]
    (doseq [value values] (put-value buffer value))
    (.flip buffer)
    {:dtype dtype :shape (mapv long (:shape array)) :buffer buffer}))

(defn write-file!
  "Write a deterministic safetensors file and atomically replace `path`.

  Tensor names are sorted, metadata keys and values must be strings, and each
  tensor's current f32/f64 dtype is preserved. Returns the destination Path."
  ([path tensors] (write-file! path tensors {} {}))
  ([path tensors metadata] (write-file! path tensors metadata {}))
  ([path tensors metadata {:keys [storage-dtype]}]
   (when-not (and (map? tensors) (seq tensors)
                  (every? string? (keys tensors)))
     (throw (ex-info "torch.safetensors: tensors must be a non-empty string-keyed map"
                     {:tensor-names (keys tensors)})))
   (when-not (and (map? metadata) (every? string? (keys metadata))
                  (every? string? (vals metadata)))
     (throw (ex-info "torch.safetensors: metadata must contain only strings"
                     {:metadata metadata})))
   (let [destination (.toAbsolutePath ^Path (if (instance? Path path)
                                               path (Path/of (str path) (make-array String 0))))
         parent (or (.getParent destination)
                    (.toAbsolutePath (Path/of "." (make-array String 0))))
         encoded (mapv (fn [[name array]]
                         [name (encode-tensor array storage-dtype)])
                       (sort-by key tensors))
         [header _]
         (reduce (fn [[entries offset] [name {:keys [dtype shape buffer]}]]
                   (let [end (+ offset (.remaining ^ByteBuffer buffer))]
                     [(assoc entries name {"dtype" dtype "shape" shape
                                           "data_offsets" [offset end]})
                      end]))
                 [(cond-> {} (seq metadata) (assoc "__metadata__" metadata)) 0]
                 encoded)
         header-bytes (padded-header-bytes header)
         temp (Files/createTempFile parent ".torch-safetensors-" ".tmp"
                                    (make-array FileAttribute 0))]
     (try
       (with-open [channel (FileChannel/open temp
                                             (into-array StandardOpenOption
                                                         [StandardOpenOption/WRITE
                                                          StandardOpenOption/TRUNCATE_EXISTING]))]
         (write-fully! channel (doto (ByteBuffer/allocate 8)
                                (.order ByteOrder/LITTLE_ENDIAN)
                                (.putLong (long (alength header-bytes)))
                                (.flip)))
         (write-fully! channel (ByteBuffer/wrap header-bytes))
         (doseq [[_ {:keys [buffer]}] encoded]
           (write-fully! channel (.duplicate ^ByteBuffer buffer)))
         (.force channel true))
       (try
         (Files/move temp destination
                     (into-array StandardCopyOption
                                 [StandardCopyOption/ATOMIC_MOVE
                                  StandardCopyOption/REPLACE_EXISTING]))
         (catch java.nio.file.AtomicMoveNotSupportedException _
           (Files/move temp destination
                       (into-array StandardCopyOption
                                   [StandardCopyOption/REPLACE_EXISTING]))))
       destination
       (finally (Files/deleteIfExists temp))))))

(defn save-weights!
  "Save model weights under stable PyTorch-compatible state-dict names."
  ([path model* weights] (save-weights! path model* weights {}))
  ([path model* weights metadata]
   (write-file! path (state/state-dict model* weights)
                (merge {"format" "pt" "torch-clj.kind" "model"} metadata))))

(defn- half->double [bits]
  (let [sign (if (zero? (bit-and bits 0x8000)) 1.0 -1.0)
        exponent (bit-and (unsigned-bit-shift-right bits 10) 0x1f)
        fraction (bit-and bits 0x3ff)]
    (cond
      (zero? exponent) (* sign (Math/pow 2.0 -14.0) (/ fraction 1024.0))
      (= exponent 31) (if (zero? fraction)
                        (* sign Double/POSITIVE_INFINITY) Double/NaN)
      :else (* sign (Math/pow 2.0 (- exponent 15.0))
               (+ 1.0 (/ fraction 1024.0))))))

(defn- decode-values [^ByteBuffer buffer dtype count]
  (mapv (fn [_]
          (double
           (case dtype
             "F16" (half->double (bit-and 0xffff (int (.getShort buffer))))
             "BF16" (Float/intBitsToFloat
                      (unchecked-int
                       (bit-shift-left (bit-and 0xffff (int (.getShort buffer))) 16)))
             "F32" (.getFloat buffer) "F64" (.getDouble buffer)
             "I8" (.get buffer) "U8" (bit-and 0xff (int (.get buffer)))
             "I16" (.getShort buffer)
             "U16" (bit-and 0xffff (int (.getShort buffer)))
             "I32" (.getInt buffer) "U32" (Integer/toUnsignedLong (.getInt buffer))
             "I64" (.getLong buffer)
             "U64" (Double/parseDouble (Long/toUnsignedString (.getLong buffer))))))
        (range count)))

(defn read-tensor
  "Read one tensor window and upload it to `backend`. `target-dtype` defaults
  to f32 so f16/bf16 checkpoints can execute through current f32 kernels."
  ([checkpoint backend name] (read-tensor checkpoint backend name :f32))
  ([checkpoint backend name target-dtype]
   (let [{:strs [dtype shape data_offsets] :as info} (tensor-info checkpoint name)]
     (when-not info
       (throw (ex-info "torch.safetensors: tensor not found"
                       {:tensor name :path (:path checkpoint)})))
     (let [[start end] data_offsets
           buffer (doto (ByteBuffer/allocate (int (- end start)))
                    (.order ByteOrder/LITTLE_ENDIAN))]
       (read-fully-at! (:channel checkpoint) buffer (+ (:data-start checkpoint) start))
       (arr/from-vec backend (decode-values buffer dtype (nelems shape))
                     (mapv long shape) target-dtype)))))

(defn load-weights
  "Load exactly the tensors required by `model*` and return its aligned weight
  vector. Strict mode rejects unrelated checkpoint tensors."
  ([path backend model*] (load-weights path backend model* {}))
  ([path backend model* {:keys [strict? dtype] :or {strict? true dtype :f32}}]
   (with-open [checkpoint (open-file path)]
     (let [required (set (map :name (state/manifest model*)))
           provided (set (tensor-names checkpoint))
           missing (vec (sort (set/difference required provided)))
           unexpected (vec (sort (set/difference provided required)))]
       (when (seq missing)
         (throw (ex-info "torch.safetensors: missing tensors" {:missing missing})))
       (when (and strict? (seq unexpected))
         (throw (ex-info "torch.safetensors: unexpected tensors"
                         {:unexpected unexpected})))
       (state/load-state-dict
        model*
        (into {} (map (fn [name]
                        [name (read-tensor checkpoint backend name dtype)]))
              required)
        {:strict? false})))))
