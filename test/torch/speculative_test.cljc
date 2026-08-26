(ns torch.speculative-test
  (:require [clojure.test :refer [deftest is]]
            [torch.speculative :as speculative]))

(deftest greedy-verification-accepts-prefix-and-target-correction
  (is (= {:tokens [1 2 0] :drafted 3 :accepted 2 :all-accepted? false}
         (speculative/verify-greedy
          [1 2 2] [[0.0 2.0 1.0] [0.0 1.0 2.0] [3.0 1.0 2.0]]))))

(deftest mtp-heads-produce-draft-and-aggregate-metrics
  (is (= [1 0 2]
         (speculative/mtp-draft [[0.0 2.0] [3.0 0.0] [0.0 1.0 4.0]])))
  (is (= {:steps 2 :drafted 5 :accepted 4 :acceptance-rate 0.8}
         (speculative/metrics [{:drafted 3 :accepted 2}
                               {:drafted 2 :accepted 2}])))
  (is (= [1 2]
         (speculative/mtp-draft-candidates
          [[[1 3.0] [0 2.0]] [[2 4.0] [3 1.0]]]
          {:temperature 0.0})))
  (is (= {:tokens [1 2 0] :drafted 3 :accepted 2 :all-accepted? false}
         (speculative/verify-greedy-tokens [1 2 2] [1 2 0]))))

(deftest stochastic-rejection-samples-positive-residual
  (let [result (speculative/verify-stochastic
                [0] [[0.9 0.1]] [[0.1 0.9]] [0.5 0.2])]
    (is (= [1] (:tokens result)))
    (is (= 0 (:accepted result)))
    (is (false? (:all-accepted? result)))))
