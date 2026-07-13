(ns torch.generate-test
  (:require [clojure.test :refer [deftest is testing]]
            [torch.generate :as generate]))

(deftest sampling-policies-are-bounded-and-deterministic
  (testing "greedy and truncation policies choose from their permitted sets"
    (is (= 1 (generate/sample-token [0.1 2.0 1.5 -0.2] {:temperature 0.0})))
    (is (= 2 (generate/sample-token [0.1 2.0 1.5 -0.2]
                                    {:top-k 2 :random-value 0.99})))
    (is (= 1 (generate/sample-token [0.1 2.0 1.5 -0.2]
                                    {:top-p 0.5 :random-value 0.99})))
    (is (= 2 (generate/sample-token [0.1 2.0 1.5 -0.2]
                                    {:temperature 0.0
                                     :previous-tokens [1]
                                     :repetition-penalty 2.0}))))
  (testing "bad policy values fail instead of silently changing semantics"
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (generate/sample-token [1.0 2.0] {:top-p 0.0})))
    (is (thrown? #?(:clj Exception :cljs js/Error)
                 (generate/sample-token [1.0 2.0] {:random-value 1.0})))))
