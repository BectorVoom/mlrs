"""The scorer API: ``make_scorer``, ``get_scorer``, ``get_scorer_names``,
``check_scoring``.

A scorer is plumbing, not arithmetic: it calls the estimator's response
method (``predict`` / ``predict_proba`` / ``decision_function``), selects the
positive-class column, and hands the result to a metric. That plumbing is
sklearn's own ``_Scorer`` class — reused so an mlrs scorer drops into
``cross_val_score``, ``GridSearchCV`` and metadata routing unchanged. What mlrs
supplies is the METRIC: every name in sklearn's scorer registry is rebuilt here
around the ``mlrs.metrics`` function of the same name, with the same sign,
response method and keyword arguments sklearn registers.
"""

import copy

__all__ = ["check_scoring", "get_scorer", "get_scorer_names", "make_scorer"]


def make_scorer(score_func, *, response_method="predict", greater_is_better=True, **kwargs):
    """Matches ``sklearn.metrics.make_scorer`` (the returned object IS a
    sklearn scorer, wrapping whatever ``score_func`` you pass — typically an
    ``mlrs.metrics`` function)."""
    from sklearn.metrics import make_scorer as _sk_make_scorer

    return _sk_make_scorer(
        score_func, response_method=response_method, greater_is_better=greater_is_better, **kwargs
    )


_REGISTRY = None


def positive_likelihood_ratio(y_true, y_pred):
    """sklearn's private scorer helper: ``LR+`` with undefined -> 1.0."""
    from ._classification import class_likelihood_ratios

    return class_likelihood_ratios(y_true, y_pred, replace_undefined_by=1.0)[0]


def negative_likelihood_ratio(y_true, y_pred):
    """sklearn's private scorer helper: ``LR-`` with undefined -> 1.0."""
    from ._classification import class_likelihood_ratios

    return class_likelihood_ratios(y_true, y_pred, replace_undefined_by=1.0)[1]


_PRIVATE = {
    "positive_likelihood_ratio": positive_likelihood_ratio,
    "negative_likelihood_ratio": negative_likelihood_ratio,
}


def _mlrs_function(name):
    from . import cluster as _cluster
    import mlrs.metrics as _m

    if name in _PRIVATE:
        return _PRIVATE[name]
    if hasattr(_m, name):
        return getattr(_m, name)
    return getattr(_cluster, name)


def _registry():
    """Mirror sklearn's ``_SCORERS`` with mlrs metric functions (built once,
    lazily, from the INSTALLED sklearn's registry so the names, signs and
    keyword arguments always agree with it)."""
    global _REGISTRY
    if _REGISTRY is None:
        from sklearn.metrics import _scorer as sk

        reg = {}
        for name, scorer in sk._SCORERS.items():
            fn = _mlrs_function(scorer._score_func.__name__)
            reg[name] = make_scorer(
                fn,
                response_method=scorer._response_method,
                greater_is_better=scorer._sign > 0,
                **scorer._kwargs,
            )
        _REGISTRY = reg
    return _REGISTRY


def get_scorer_names():
    """Matches ``sklearn.metrics.get_scorer_names``."""
    return sorted(_registry().keys())


def get_scorer(scoring):
    """Matches ``sklearn.metrics.get_scorer``: a string returns a fresh copy of
    the registered mlrs scorer; a callable (or ``None``) is returned as is."""
    if isinstance(scoring, str):
        try:
            return copy.deepcopy(_registry()[scoring])
        except KeyError:
            raise ValueError(
                "%r is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get "
                "valid options." % (scoring,)
            ) from None
    return scoring


def check_scoring(estimator=None, scoring=None, *, allow_none=False, raise_exc=True):
    """Matches ``sklearn.metrics.check_scoring``. String scorers (alone, or in a
    list/tuple/set/dict of them) resolve to the mlrs registry; everything else
    follows sklearn's own rules."""
    from sklearn.metrics import check_scoring as _sk_check_scoring

    if isinstance(scoring, str):
        return get_scorer(scoring)
    if isinstance(scoring, (list, tuple, set)) and all(isinstance(s, str) for s in scoring):
        from sklearn.metrics._scorer import _MultimetricScorer

        return _MultimetricScorer(scorers={s: get_scorer(s) for s in scoring}, raise_exc=raise_exc)
    if isinstance(scoring, dict) and all(isinstance(v, str) for v in scoring.values()):
        from sklearn.metrics._scorer import _MultimetricScorer

        return _MultimetricScorer(scorers={k: get_scorer(v) for k, v in scoring.items()}, raise_exc=raise_exc)
    return _sk_check_scoring(estimator, scoring, allow_none=allow_none, raise_exc=raise_exc)
