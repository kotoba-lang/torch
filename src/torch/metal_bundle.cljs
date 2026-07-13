(ns torch.metal-bundle
  "Reader for TGBNDL1 manifest + binary tensor payload containers."
  (:require [cljs.reader :as reader]))

(def magic "TGBNDL1\n")

(defn- u32-le [view offset]
  (.getUint32 view offset true))

(defn- payload-values [bytes payload-base encoding {:keys [offset length]}]
  (let [start (+ payload-base offset)
        end (+ start length)]
    (when-not (and (int? offset) (<= 0 offset) (int? length) (<= 0 length)
                   (<= payload-base start end (.-byteLength bytes))
                   (or (not= encoding :f32-le) (zero? (mod length 4))))
      (throw (js/Error. "bundle tensor payload is out of bounds or misaligned")))
    (case encoding
      :u8 (mapv #(aget bytes %) (range start (+ start length)))
      :f32-le
      (let [view (js/DataView. (.-buffer bytes) (+ (.-byteOffset bytes) start) length)]
        (mapv #(.getFloat32 view (* 4 %) true) (range (quot length 4))))
      (throw (js/Error. "unsupported bundle tensor encoding")))))

(defn load-bundle
  "Synchronously read and validate a compact bundle from Deno filesystem."
  [path]
  (let [bytes (js/Deno.readFileSync path)
        decoder (js/TextDecoder.)
        _ (when (< (.-byteLength bytes) 12)
            (throw (js/Error. "truncated TGBNDL1 header")))
        prefix (.decode decoder (.slice bytes 0 8))]
    (when-not (= magic prefix)
      (throw (js/Error. "invalid TGBNDL1 bundle magic")))
    (let [view (js/DataView. (.-buffer bytes) (.-byteOffset bytes) (.-byteLength bytes))
          manifest-length (u32-le view 8)
          manifest-start 12
          payload-base (+ manifest-start manifest-length)
          _ (when (> payload-base (.-byteLength bytes))
              (throw (js/Error. "bundle manifest is outside file bounds")))
          manifest (reader/read-string
                    (.decode decoder (.slice bytes manifest-start payload-base)))]
      (when-not (= :torch/gguf-metal-bundle-v2 (:format manifest))
        (throw (js/Error. "unsupported Metal bundle version")))
      (update manifest :weights
              (fn [weights]
                (mapv (fn [entry]
                        (into {}
                              (map (fn [[key descriptor]]
                                     (let [values (payload-values
                                                   bytes payload-base
                                                   (:encoding descriptor)
                                                   (:payload descriptor))
                                           _ (when (and (= :f32-le (:encoding descriptor))
                                                        (not= (count values)
                                                              (reduce * 1 (:shape descriptor))))
                                               (throw (js/Error.
                                                       "dense bundle tensor shape mismatch")))]
                                       [key (-> descriptor
                                                (dissoc :payload :encoding)
                                                (assoc (if (= :u8 (:encoding descriptor))
                                                         :bytes :values)
                                                       values))])))
                              entry))
                      weights))))))
