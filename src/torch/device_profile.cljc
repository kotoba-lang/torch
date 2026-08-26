(ns torch.device-profile
  "Validated accelerator profiles used by native and distributed inference."
  (:require [clojure.string :as str]))

(defn classify
  "Classify a WebGPU adapter from its public information map/string."
  [adapter]
  (let [s (str/lower-case (str adapter))]
    (cond
      (or (str/includes? s "llvmpipe") (str/includes? s "software")) :software
      (and (str/includes? s "intel")
           (or (str/includes? s "bmg") (str/includes? s "b70")
               (str/includes? s "e223"))) :intel-arc-b70-vulkan
      (or (str/includes? s "apple") (str/includes? s "metal")) :apple-metal
      (or (str/includes? s "vulkan") (str/includes? s "discretegpu")) :vulkan-gpu
      :else :unknown)))

(def profiles
  {:intel-arc-b70-vulkan
   {:backend :webgpu-vulkan
    :collective :host-staged
    :peer-to-peer? false
    :preferred-features #{:shader-f16}
    :device-ids #{0xe223}}
   :apple-metal
   {:backend :webgpu-metal :collective :shared-memory
    :peer-to-peer? false :preferred-features #{:shader-f16}}
   :vulkan-gpu
   {:backend :webgpu-vulkan :collective :host-staged
    :peer-to-peer? false :preferred-features #{:shader-f16}}})

(defn profile [adapter]
  (let [kind (classify adapter)]
    (assoc (get profiles kind {:backend :unsupported}) :kind kind)))

(defn validate!
  ([adapter] (validate! adapter nil))
  ([adapter expected]
   (let [{:keys [kind backend] :as p} (profile adapter)]
     (when (= :software kind)
       (throw (ex-info "software WebGPU adapters are not inference devices"
                       {:adapter adapter :kind kind})))
     (when (= :unsupported backend)
       (throw (ex-info "unsupported WebGPU adapter" {:adapter adapter})))
     (when (and expected (not= expected kind))
       (throw (ex-info "WebGPU adapter does not match requested device"
                       {:expected expected :actual kind :adapter adapter})))
     p)))
