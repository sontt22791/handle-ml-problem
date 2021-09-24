# handle-ml-problem






# feature selection

- tham khảo: https://www.kaggle.com/arthurtok/feature-ranking-rfe-random-forest-linear-models#INTRODUCTION

## RFE
- explain: https://www.linkedin.com/pulse/what-recursive-feature-elimination-amit-mittal/ 
    - từ bài này mình có 1 câu hỏi: sao tác giả tách backward elimination và RFE??? chúng có khác nhau k??? => ko khác nhau: https://stats.stackexchange.com/questions/450518/rfe-vs-backward-elimination-is-there-a-difference
- how RFECV works?: https://www.quora.com/scikit-learn-How-does-RFECV-make-use-of-cross-validation
- sklearn:
    - user guide => explain rfe và rfecv để làm gì:https://scikit-learn.org/stable/modules/feature_selection.html#rfe
    - vd sử dụng rfecv để tìm optimal n_features: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

# Hyperparameter optimization

- GridsearchCV & RandomSearchCV => 
- `scikit-optimize (skopt)` uses `Bayesian optimization` with `gaussian process` can be accomplished by using `gp_minimize function`
- `hyperopt` uses `Tree-structured Parzen Estimator (TPE)` to find the most optimal parameters
=> `optimization function` return `negative accuracy` cause we cannot minimize the accuracy, but we can minimize it when we multiply it by -1.
![](screenshot/table-hyperparameters.png)

- tham khảo: https://neptune.ai/blog/hyperparameter-tuning-on-any-python-script

