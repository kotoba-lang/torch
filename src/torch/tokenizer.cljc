(ns torch.tokenizer
  "Portable BPE tokenizer with GGUF-style `<0xHH>` UTF-8 byte fallback."
  (:require [clojure.string :as str])
  #?(:clj (:import [java.nio.charset StandardCharsets])))

(defn- utf8-bytes [s]
  #?(:clj (mapv #(bit-and (int %) 0xff)
                (.getBytes ^String s StandardCharsets/UTF_8))
     :cljs (vec (.encode (js/TextEncoder.) s))))

(defn- bytes->utf8 [bytes]
  #?(:clj (String. (byte-array (map #(unchecked-byte (int %)) bytes))
                   StandardCharsets/UTF_8)
     :cljs (.decode (js/TextDecoder. "utf-8" #js {:fatal false})
                    (js/Uint8Array. (clj->js bytes)))))

(defn- byte-token [byte]
  (str "<0x" (str/upper-case
                #?(:clj (format "%02x" (int byte))
                   :cljs (.padStart (.toString byte 16) 2 "0"))) ">"))

(def byte-token-pattern #"^<0x([0-9A-Fa-f]{2})>$")

(defn tokenizer
  "Build a tokenizer from `:tokens` (ID order) and ordered `:merges` pairs.
  Optional keys: `:unk-id`, `:bos-id`, `:eos-id`, `:add-bos?`, `:add-eos?`,
  and `:space-prefix` (for example `\"▁\"` for SentencePiece-style spaces)."
  [{:keys [tokens merges] :as options}]
  (when-not (and (vector? tokens) (every? string? tokens))
    (throw (ex-info "tokenizer :tokens must be a vector of strings" {})))
  (let [token->id (reduce-kv (fn [m id token]
                               (if (contains? m token) m (assoc m token id)))
                             {} tokens)
        merge-ranks (into {} (map-indexed (fn [rank pair] [(vec pair) rank]) merges))]
    (assoc options :tokens tokens :token->id token->id :merge-ranks merge-ranks)))

(defn- codepoints [text]
  #?(:clj (mapv (fn [cp] (String. (Character/toChars cp)))
                (.toArray (.codePoints ^String text)))
     :cljs (vec (js/Array.from text))))

(defn- initial-symbols [{:keys [token->id]} text]
  (mapcat (fn [character]
            (if (contains? token->id character)
              [character]
              (let [bytes (mapv byte-token (utf8-bytes character))]
                (if (every? #(contains? token->id %) bytes)
                  bytes [character]))))
          (codepoints text)))

(defn- best-merge [symbols merge-ranks]
  (->> (partition 2 1 symbols)
       (map-indexed (fn [index pair]
                      (when-let [rank (get merge-ranks (vec pair))]
                        [rank index pair])))
       (remove nil?)
       (sort-by (juxt first second))
       first))

(defn- merge-at [symbols index]
  (vec (concat (subvec symbols 0 index)
               [(str (nth symbols index) (nth symbols (inc index)))]
               (subvec symbols (+ index 2)))))

(defn encode
  "Encode text to token IDs. Unknown codepoints fall back to their UTF-8 byte
  tokens; if a required byte token is absent, `:unk-id` is used or an error is
  thrown."
  [tokenizer text]
  (let [{:keys [token->id merge-ranks space-prefix bos-id eos-id add-bos? add-eos?
                unk-id]} tokenizer
        text (if space-prefix
               (str/replace text " " space-prefix) text)
        initial (vec (initial-symbols tokenizer text))
        symbols (loop [symbols initial]
                  (if-let [[_rank index _pair] (best-merge symbols merge-ranks)]
                    (recur (merge-at symbols index)) symbols))
        encoded (mapv (fn [symbol]
                        (or (get token->id symbol) unk-id
                            (throw (ex-info "tokenizer vocabulary lacks encoded symbol"
                                            {:symbol symbol}))))
                      symbols)
        ids (if add-bos? (into [bos-id] encoded) encoded)]
    (if add-eos? (conj ids eos-id) ids)))

(defn decode
  "Decode token IDs. BOS/EOS and any IDs in `:special-ids` are skipped by
  default; pass `{:skip-special? false}` to preserve their token strings."
  ([tokenizer ids] (decode tokenizer ids {}))
  ([{:keys [tokens bos-id eos-id special-ids space-prefix] :as _tokenizer}
    ids {:keys [skip-special?] :or {skip-special? true}}]
   (let [specials (set (concat special-ids [bos-id eos-id]))
         bytes (mapcat (fn [id]
                         (when-not (and skip-special? (contains? specials id))
                           (let [token (nth tokens id nil)]
                             (when-not token
                               (throw (ex-info "token ID is outside vocabulary" {:id id})))
                             (if-let [[_ hex] (re-matches byte-token-pattern token)]
                               [#?(:clj (Integer/parseInt hex 16)
                                   :cljs (js/parseInt hex 16))]
                               (utf8-bytes token)))))
                       ids)
         text (bytes->utf8 (vec bytes))]
     (if space-prefix (str/replace text space-prefix " ") text))))

(defn decode-step
  "Incremental decode state. Returns `{:state ... :text newly-emitted-text}`.
  Re-decoding the short generated prefix also handles UTF-8 byte tokens split
  across calls without exposing replacement characters as committed output."
  ([tokenizer token-id] (decode-step tokenizer nil token-id))
  ([tokenizer state token-id]
   (let [ids (conj (vec (:ids state)) token-id)
         decoded (decode tokenizer ids)
         previous (or (:complete state) "")
         complete (if (str/includes? decoded "�") previous decoded)
         prefix-length (loop [n (min (count previous) (count complete))]
                         (if (or (zero? n) (= (subs previous 0 n) (subs complete 0 n)))
                           n (recur (dec n))))]
     {:state {:ids ids :complete complete}
      :text (subs complete prefix-length)})))
