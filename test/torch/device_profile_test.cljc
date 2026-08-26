(ns torch.device-profile-test
  (:require [clojure.test :refer [deftest is testing]]
            [torch.device-profile :as device]))

(deftest recognizes-and-validates-intel-arc-pro-b70
  (let [p (device/validate! "Intel(R) Graphics BMG G31 (0xE223), Vulkan"
                            :intel-arc-b70-vulkan)]
    (is (= :webgpu-vulkan (:backend p)))
    (is (= :host-staged (:collective p)))
    (is (false? (:peer-to-peer? p)))))

(deftest rejects-software-and-device-mismatch
  (testing "llvmpipe never silently becomes a GPU"
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (device/validate! "Vulkan llvmpipe software adapter"))))
  (testing "explicit B70 selection is fail closed"
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (device/validate! "Apple M4 Metal" :intel-arc-b70-vulkan)))))
