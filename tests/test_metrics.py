import math

from src.evaluation.metrics import recall_at_k, average_precision_at_k, ndcg_at_k

# Toy examples with hand-computed expected values to guarantee the metrics are correct.
# Игрушечные примеры с вручную посчитанными ответами, чтобы гарантировать корректность метрик.


def test_recall_hit_and_miss():
    # One relevant item present in top-k -> recall 1.0; absent -> 0.0
    assert recall_at_k(["a", "b", "c"], {"b"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"z"}, 3) == 0.0


def test_recall_multiple_relevant():
    # 1 of 2 relevant items in top-3 -> 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5


def test_average_precision():
    # relevant at rank 2 (index 1) -> AP = (1/2) / 1 = 0.5
    assert average_precision_at_k(["a", "b", "c"], {"b"}, 3) == 0.5
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2 = 0.8333...
    assert math.isclose(average_precision_at_k(["a", "b", "c"], {"a", "c"}, 3),
                        (1.0 + 2.0 / 3.0) / 2.0)


def test_ndcg():
    # relevant at rank 1 -> perfect ranking -> NDCG = 1.0
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    # relevant at rank 3 (index 2) -> DCG = 1/log2(4) = 0.5, IDCG = 1 -> 0.5
    assert math.isclose(ndcg_at_k(["a", "b", "c"], {"c"}, 3), 0.5)


def test_empty_relevant():
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert average_precision_at_k(["a"], set(), 3) == 0.0
    assert ndcg_at_k(["a"], set(), 3) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "passed")
    print("all metric tests passed")
