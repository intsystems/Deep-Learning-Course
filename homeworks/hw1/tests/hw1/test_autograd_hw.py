import sys

sys.path.append("./python")
sys.path.append("./apps")
from simple_ml import *
import numdifftools as nd

import numpy as np
import needle as ndl


def assert_allclose_list(list1, list2, rtol=1e-05, atol=1e-08):
    assert len(list1) == len(list2)
    assert all(
        np.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(list1, list2)
    ), "Arrays are not close!"


##############################################################################
### TESTS/SUBMISSION CODE FOR forward passes
def test_power_scalar_forward():
    np.testing.assert_allclose(
        ndl.power_scalar(ndl.Tensor([[0.5, 2.0, 3.0]]), scalar=2).numpy(),
        np.array([[0.25, 4.0, 9.0]]),
    )


def test_divide_forward():
    np.testing.assert_allclose(
        ndl.divide(
            ndl.Tensor([[3.3, 4.35, 1.2], [2.45, 0.95, 2.55]]),
            ndl.Tensor([[4.6, 4.35, 4.8], [0.65, 0.7, 4.4]]),
        ).numpy(),
        np.array(
            [
                [0.717391304348, 1.0, 0.25],
                [3.769230769231, 1.357142857143, 0.579545454545],
            ]
        ),
    )


def test_divide_scalar_forward():
    np.testing.assert_allclose(
        ndl.divide_scalar(ndl.Tensor([[1.7, 1.45]]), scalar=12).numpy(),
        np.array([[0.141666666667, 0.120833333333]]),
    )


def test_matmul_forward():
    np.testing.assert_allclose(
        ndl.matmul(
            ndl.Tensor([[4.95, 1.75, 0.25], [4.15, 4.25, 0.3], [0.3, 0.4, 2.1]]),
            ndl.Tensor([[1.35, 2.2, 1.55], [3.85, 4.8, 2.6], [1.15, 0.85, 4.15]]),
        ).numpy(),
        np.array(
            [[13.7075, 19.5025, 13.26], [22.31, 29.785, 18.7275], [4.36, 4.365, 10.22]]
        ),
    )
    np.testing.assert_allclose(
        ndl.matmul(
            ndl.Tensor([[3.8, 0.05], [2.3, 3.35], [1.6, 2.6]]),
            ndl.Tensor([[1.1, 3.5, 3.7], [0.05, 1.25, 1.0]]),
        ).numpy(),
        np.array(
            [[4.1825, 13.3625, 14.11], [2.6975, 12.2375, 11.86], [1.89, 8.85, 8.52]]
        ),
    )
    np.testing.assert_allclose(
        ndl.matmul(
            ndl.Tensor(
                [
                    [[4.0, 2.15], [1.25, 1.35], [0.75, 1.6]],
                    [[2.9, 2.15], [3.3, 4.1], [2.5, 0.25]],
                    [[2.9, 4.35], [1.2, 3.5], [3.55, 3.95]],
                    [[2.55, 4.35], [4.25, 0.2], [3.95, 3.4]],
                    [[2.2, 2.05], [0.95, 1.8], [2.7, 2.0]],
                    [[0.45, 1.1], [3.15, 0.7], [2.9, 1.95]],
                ]
            ),
            ndl.Tensor(
                [
                    [[2.7, 4.05, 0.1], [1.75, 3.05, 2.3]],
                    [[0.55, 4.1, 2.3], [4.45, 2.35, 2.55]],
                    [[1.2, 3.95, 4.6], [4.2, 3.5, 3.35]],
                    [[2.55, 4.4, 2.05], [2.4, 0.6, 4.65]],
                    [[2.95, 0.8, 0.6], [0.45, 1.3, 0.75]],
                    [[1.25, 2.1, 0.4], [0.85, 3.5, 3.7]],
                ]
            ),
        ).numpy(),
        np.array(
            [
                [
                    [14.5625, 22.7575, 5.345],
                    [5.7375, 9.18, 3.23],
                    [4.825, 7.9175, 3.755],
                ],
                [
                    [11.1625, 16.9425, 12.1525],
                    [20.06, 23.165, 18.045],
                    [2.4875, 10.8375, 6.3875],
                ],
                [
                    [21.75, 26.68, 27.9125],
                    [16.14, 16.99, 17.245],
                    [20.85, 27.8475, 29.5625],
                ],
                [
                    [16.9425, 13.83, 25.455],
                    [11.3175, 18.82, 9.6425],
                    [18.2325, 19.42, 23.9075],
                ],
                [[7.4125, 4.425, 2.8575], [3.6125, 3.1, 1.92], [8.865, 4.76, 3.12]],
                [[1.4975, 4.795, 4.25], [4.5325, 9.065, 3.85], [5.2825, 12.915, 8.375]],
            ]
        ),
    )
    np.testing.assert_allclose(
        ndl.matmul(
            ndl.Tensor([[1.9, 1.9], [4.8, 4.9], [3.25, 3.75]]),
            ndl.Tensor(
                [
                    [[1.25, 1.8, 1.95], [3.75, 2.85, 2.25]],
                    [[1.75, 2.7, 3.3], [2.95, 1.55, 3.85]],
                    [[4.2, 3.05, 3.35], [3.3, 4.75, 2.1]],
                ]
            ),
        ).numpy(),
        np.array(
            [
                [
                    [9.5, 8.835, 7.98],
                    [24.375, 22.605, 20.385],
                    [18.125, 16.5375, 14.775],
                ],
                [
                    [8.93, 8.075, 13.585],
                    [22.855, 20.555, 34.705],
                    [16.75, 14.5875, 25.1625],
                ],
                [
                    [14.25, 14.82, 10.355],
                    [36.33, 37.915, 26.37],
                    [26.025, 27.725, 18.7625],
                ],
            ]
        ),
    )
    np.testing.assert_allclose(
        ndl.matmul(
            ndl.Tensor(
                [
                    [[3.4, 2.95], [0.25, 1.95], [4.4, 4.4]],
                    [[0.55, 1.1], [0.75, 1.55], [4.1, 1.2]],
                    [[1.5, 4.05], [1.5, 1.55], [2.3, 1.25]],
                ]
            ),
            ndl.Tensor([[2.2, 0.65, 2.5], [2.5, 1.3, 0.15]]),
        ).numpy(),
        np.array(
            [
                [
                    [14.855, 6.045, 8.9425],
                    [5.425, 2.6975, 0.9175],
                    [20.68, 8.58, 11.66],
                ],
                [[3.96, 1.7875, 1.54], [5.525, 2.5025, 2.1075], [12.02, 4.225, 10.43]],
                [[13.425, 6.24, 4.3575], [7.175, 2.99, 3.9825], [8.185, 3.12, 5.9375]],
            ]
        ),
    )


def test_summation_forward():
    np.testing.assert_allclose(
        ndl.summation(
            ndl.Tensor(
                [
                    [2.2, 4.35, 1.4, 0.3, 2.65],
                    [1.0, 0.85, 2.75, 3.8, 1.55],
                    [3.2, 2.3, 3.45, 0.7, 0.0],
                ]
            )
        ).numpy(),
        np.array(30.5),
    )
    np.testing.assert_allclose(
        ndl.summation(
            ndl.Tensor(
                [
                    [1.05, 2.55, 1.0],
                    [2.95, 3.7, 2.6],
                    [0.1, 4.1, 3.3],
                    [1.1, 3.4, 3.4],
                    [1.8, 4.55, 2.3],
                ]
            ),
            axes=1,
        ).numpy(),
        np.array([4.6, 9.25, 7.5, 7.9, 8.65]),
    )
    np.testing.assert_allclose(
        ndl.summation(
            ndl.Tensor([[1.5, 3.85, 3.45], [1.35, 1.3, 0.65], [2.6, 4.55, 0.25]]),
            axes=0,
        ).numpy(),
        np.array([5.45, 9.7, 4.35]),
    )


def test_broadcast_to_forward():
    np.testing.assert_allclose(
        ndl.broadcast_to(ndl.Tensor([[1.85, 0.85, 0.6]]), shape=(3, 3, 3)).numpy(),
        np.array(
            [
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
                [[1.85, 0.85, 0.6], [1.85, 0.85, 0.6], [1.85, 0.85, 0.6]],
            ]
        ),
    )


def test_reshape_forward():
    np.testing.assert_allclose(
        ndl.reshape(
            ndl.Tensor(
                [
                    [2.9, 2.0, 2.4],
                    [3.95, 3.95, 4.65],
                    [2.1, 2.5, 2.7],
                    [1.9, 4.85, 3.25],
                    [3.35, 3.45, 3.45],
                ]
            ),
            shape=(15,),
        ).numpy(),
        np.array(
            [
                2.9,
                2.0,
                2.4,
                3.95,
                3.95,
                4.65,
                2.1,
                2.5,
                2.7,
                1.9,
                4.85,
                3.25,
                3.35,
                3.45,
                3.45,
            ]
        ),
    )
    np.testing.assert_allclose(
        ndl.reshape(
            ndl.Tensor(
                [
                    [[4.1, 4.05, 1.35, 1.65], [3.65, 0.9, 0.65, 4.15]],
                    [[4.7, 1.4, 2.55, 4.8], [2.8, 1.75, 2.8, 0.6]],
                    [[3.75, 0.6, 0.0, 3.5], [0.15, 1.9, 4.75, 2.8]],
                ]
            ),
            shape=(2, 3, 4),
        ).numpy(),
        np.array(
            [
                [
                    [4.1, 4.05, 1.35, 1.65],
                    [3.65, 0.9, 0.65, 4.15],
                    [4.7, 1.4, 2.55, 4.8],
                ],
                [[2.8, 1.75, 2.8, 0.6], [3.75, 0.6, 0.0, 3.5], [0.15, 1.9, 4.75, 2.8]],
            ]
        ),
    )


def test_negate_forward():
    np.testing.assert_allclose(
        ndl.negate(ndl.Tensor([[1.45, 0.55]])).numpy(), np.array([[-1.45, -0.55]])
    )


def test_transpose_forward():
    np.testing.assert_allclose(
        ndl.transpose(ndl.Tensor([[[1.95]], [[2.7]], [[3.75]]]), axes=(1, 2)).numpy(),
        np.array([[[1.95]], [[2.7]], [[3.75]]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor([[[[0.95]]], [[[2.55]]], [[[0.45]]]]), axes=(2, 3)
        ).numpy(),
        np.array([[[[0.95]]], [[[2.55]]], [[[0.45]]]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor(
                [
                    [[[0.4, 0.05], [2.95, 1.3]], [[4.8, 1.2], [1.65, 3.1]]],
                    [[[1.45, 3.05], [2.25, 0.1]], [[0.45, 4.75], [1.5, 1.8]]],
                    [[[1.5, 4.65], [1.35, 2.7]], [[2.0, 1.65], [2.05, 1.2]]],
                ]
            )
        ).numpy(),
        np.array(
            [
                [[[0.4, 2.95], [0.05, 1.3]], [[4.8, 1.65], [1.2, 3.1]]],
                [[[1.45, 2.25], [3.05, 0.1]], [[0.45, 1.5], [4.75, 1.8]]],
                [[[1.5, 1.35], [4.65, 2.7]], [[2.0, 2.05], [1.65, 1.2]]],
            ]
        ),
    )
    np.testing.assert_allclose(
        ndl.transpose(ndl.Tensor([[[2.45]], [[3.5]], [[0.9]]]), axes=(0, 1)).numpy(),
        np.array([[[2.45], [3.5], [0.9]]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(ndl.Tensor([[4.4, 2.05], [1.85, 2.25], [0.15, 1.4]])).numpy(),
        np.array([[4.4, 1.85, 0.15], [2.05, 2.25, 1.4]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor([[0.05, 3.7, 1.35], [4.45, 3.25, 1.95], [2.45, 4.4, 4.5]])
        ).numpy(),
        np.array([[0.05, 4.45, 2.45], [3.7, 3.25, 4.4], [1.35, 1.95, 4.5]]),
    )
    np.testing.assert_allclose(
        ndl.transpose(
            ndl.Tensor(
                [
                    [[0.55, 1.8, 0.2], [0.8, 2.75, 3.7], [0.95, 1.4, 0.8]],
                    [[0.75, 1.6, 1.35], [3.75, 4.0, 4.55], [1.85, 2.5, 4.8]],
                    [[0.2, 3.35, 3.4], [0.3, 4.85, 4.85], [4.35, 4.25, 3.05]],
                ]
            ),
            axes=(0, 1),
        ).numpy(),
        np.array(
            [
                [[0.55, 1.8, 0.2], [0.75, 1.6, 1.35], [0.2, 3.35, 3.4]],
                [[0.8, 2.75, 3.7], [3.75, 4.0, 4.55], [0.3, 4.85, 4.85]],
                [[0.95, 1.4, 0.8], [1.85, 2.5, 4.8], [4.35, 4.25, 3.05]],
            ]
        ),
    )


def test_log_forward():
    np.testing.assert_allclose(
        ndl.log(ndl.Tensor([[4.0], [4.55]])).numpy(),
        np.array([[1.38629436112], [1.515127232963]]),
    )


def test_exp_forward():
    np.testing.assert_allclose(
        ndl.exp(ndl.Tensor([[4.0], [4.55]])).numpy(),
        np.array([[54.59815003], [94.63240831]]),
    )


def test_forward_2():
    # Test case 1
    result = ndl.power_scalar(ndl.Tensor([[0.3, 2.5]]), scalar=3).numpy()
    np.testing.assert_allclose(result, np.array([[0.027, 15.625]]))

    # Test case 2
    result = ndl.divide(
        ndl.Tensor([[3.4, 2.35, 1.25], [0.45, 1.95, 2.55]]),
        ndl.Tensor([[4.9, 4.35, 4.1], [0.65, 0.7, 4.04]]),
    ).numpy()
    np.testing.assert_allclose(
        result,
        np.array(
            [[0.69387755, 0.54022989, 0.30487805], [0.69230769, 2.78571429, 0.63118812]]
        ),
    )

    # Test case 3
    result = ndl.divide_scalar(ndl.Tensor([[1.4, 2.89]]), scalar=7).numpy()
    np.testing.assert_allclose(result, np.array([[0.2, 0.41285714]]))

    # Test case 4
    result = ndl.matmul(
        ndl.Tensor([[1.75, 1.75, 0.25], [4.95, 4.35, 0.3], [0.3, 1.4, 2.1]]),
        ndl.Tensor([[2.35, 2.2, 1.85], [7.85, 4.88, 2.6], [1.15, 0.25, 4.19]]),
    ).numpy()
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [18.1375, 12.4525, 8.835],
                [46.125, 32.193, 21.7245],
                [14.11, 8.017, 12.994],
            ]
        ),
    )

    # Test case 5
    result = ndl.summation(
        ndl.Tensor(
            [
                [1.2, 4.35, 1.4, 0.3, 0.75],
                [2.0, 1.85, 7.75, 3.7, 1.55],
                [9.2, 2.3, 3.45, 0.7, 0.0],
            ]
        )
    ).numpy()
    np.testing.assert_allclose(result, np.array(40.5))

    # Test case 6
    result = ndl.summation(
        ndl.Tensor(
            [
                [5.05, 2.55, 1.0],
                [2.75, 3.7, 2.1],
                [0.1, 4.1, 3.3],
                [1.4, 0.4, 3.4],
                [2.8, 0.55, 2.9],
            ]
        ),
        axes=1,
    ).numpy()
    np.testing.assert_allclose(result, np.array([8.6, 8.55, 7.5, 5.2, 6.25]))

    # Test case 7
    result = ndl.broadcast_to(ndl.Tensor([[1.95, 3.85, -0.6]]), shape=(3, 3, 3)).numpy()
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [[1.95, 3.85, -0.6], [1.95, 3.85, -0.6], [1.95, 3.85, -0.6]],
                [[1.95, 3.85, -0.6], [1.95, 3.85, -0.6], [1.95, 3.85, -0.6]],
                [[1.95, 3.85, -0.6], [1.95, 3.85, -0.6], [1.95, 3.85, -0.6]],
            ]
        ),
    )

    # Test case 8
    result = ndl.reshape(
        ndl.Tensor(
            [
                [7.9, 2.0, 2.4],
                [3.11, 3.95, 0.65],
                [2.1, 2.18, 2.2],
                [1.9, 4.54, 3.25],
                [1.35, 7.45, 3.45],
            ]
        ),
        shape=(15,),
    ).numpy()
    np.testing.assert_allclose(
        result,
        np.array(
            [
                7.9,
                2.0,
                2.4,
                3.11,
                3.95,
                0.65,
                2.1,
                2.18,
                2.2,
                1.9,
                4.54,
                3.25,
                1.35,
                7.45,
                3.45,
            ]
        ),
    )

    # Test case 9
    result = ndl.reshape(
        ndl.Tensor(
            [
                [[5.1, 4.05, 1.25, 4.65], [3.65, 0.9, 0.65, 1.65]],
                [[4.7, 1.4, 2.55, 4.8], [2.8, 1.75, 3.8, 0.6]],
                [[3.75, 0.6, 1.0, 3.5], [8.15, 1.9, 4.55, 2.83]],
            ]
        ),
        shape=(2, 3, 4),
    ).numpy()
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [
                    [5.1, 4.05, 1.25, 4.65],
                    [3.65, 0.9, 0.65, 1.65],
                    [4.7, 1.4, 2.55, 4.8],
                ],
                [[2.8, 1.75, 3.8, 0.6], [3.75, 0.6, 1.0, 3.5], [8.15, 1.9, 4.55, 2.83]],
            ]
        ),
    )

    # Test case 10
    result = ndl.negate(ndl.Tensor([[1.45, 0.55]])).numpy()
    np.testing.assert_allclose(result, np.array([[-1.45, -0.55]]))

    # Test case 11
    result = ndl.transpose(
        ndl.Tensor([[[3.45]], [[2.54]], [[1.91]]]), axes=(0, 1)
    ).numpy()
    np.testing.assert_allclose(result, np.array([[[3.45], [2.54], [1.91]]]))

    # Test case 12
    result = ndl.transpose(
        ndl.Tensor([[4.45, 2.15], [1.89, 1.21], [6.15, 2.42]])
    ).numpy()
    np.testing.assert_allclose(
        result, np.array([[4.45, 1.89, 6.15], [2.15, 1.21, 2.42]])
    )

    # Test case 13
    result = ndl.log(ndl.Tensor([[[3.45]], [[2.54]], [[1.91]]])).numpy()
    np.testing.assert_allclose(
        result, np.array([[[1.23837423]], [[0.93216408]], [[0.64710324]]])
    )

    # Test case 14
    result = ndl.log(ndl.Tensor([[4.45, 2.15], [1.89, 1.21], [6.15, 2.42]])).numpy()
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1.4929041, 0.76546784],
                [0.63657683, 0.19062036],
                [1.81645208, 0.88376754],
            ]
        ),
    )

    # Test case 15
    result = ndl.exp(ndl.Tensor([[[3.45]], [[2.54]], [[1.91]]])).numpy()
    np.testing.assert_allclose(
        result, np.array([[[31.50039231]], [[12.67967097]], [[6.7530888]]])
    )

    # Test case 16
    result = ndl.exp(ndl.Tensor([[4.45, 2.15], [1.89, 1.21], [6.15, 2.42]])).numpy()
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [85.626944, 8.5848584],
                [6.61936868, 3.35348465],
                [468.71738678, 11.24585931],
            ]
        ),
    )


##############################################################################
### TESTS/SUBMISSION CODE FOR backward passes


def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.numpy()
            for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_power_scalar_backward():
    gradient_check(
        ndl.power_scalar, ndl.Tensor(np.random.randn(5, 4)), scalar=np.random.randint(1)
    )


def test_divide_backward():
    gradient_check(
        ndl.divide,
        ndl.Tensor(np.random.randn(5, 4)),
        ndl.Tensor(5 + np.random.randn(5, 4)),
    )


def test_divide_scalar_backward():
    gradient_check(
        ndl.divide_scalar, ndl.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1)
    )


def test_matmul_simple_backward():
    gradient_check(
        ndl.matmul, ndl.Tensor(np.random.randn(5, 4)), ndl.Tensor(np.random.randn(4, 5))
    )


def test_matmul_batched_backward():
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(6, 6, 5, 4)),
        ndl.Tensor(np.random.randn(6, 6, 4, 3)),
    )
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(6, 6, 5, 4)),
        ndl.Tensor(np.random.randn(4, 3)),
    )
    gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(5, 4)),
        ndl.Tensor(np.random.randn(6, 6, 4, 3)),
    )


def test_reshape_backward():
    gradient_check(ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(4, 5))


def test_negate_backward():
    gradient_check(ndl.negate, ndl.Tensor(np.random.randn(5, 4)))


def test_transpose_backward():
    gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
    gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(0, 1))


def test_broadcast_to_backward():
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(
        ndl.broadcast_to,
        ndl.Tensor(
            np.random.randn(
                1,
            )
        ),
        shape=(3, 3, 3),
    )
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(
        ndl.broadcast_to, ndl.Tensor(np.random.randn(5, 4, 1)), shape=(5, 4, 3)
    )


def test_summation_backward():
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5, 4)), axes=(1,))
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5, 4)), axes=(0,))
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5, 4)), axes=(0, 1))
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5, 4, 1)), axes=(0, 1))


def test_log_backward():
    gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5, 4)))


def test_exp_backward():
    gradient_check(ndl.exp, ndl.Tensor(1 + np.random.rand(5, 4)))


def test_backward_2():
    np.random.seed(0)

    # Test case 1
    result = gradient_check(
        ndl.power_scalar, ndl.Tensor(np.random.randn(3, 5)), scalar=np.random.randint(1)
    )
    np.testing.assert_allclose(
        result,
        [
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.0, 0.0, -0.0, -0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ],
    )

    # Test case 2
    result = gradient_check(
        ndl.divide,
        ndl.Tensor(np.random.randn(3, 5)),
        ndl.Tensor(6 + np.random.randn(3, 5)),
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [0.16247092, 0.15678497, 0.19560996, 0.24880551, 0.17692577],
                [0.16243394, 0.13830703, 0.138843, 0.17816822, 0.1755095],
                [0.20196116, 0.21834147, 0.23289775, 0.1257739, 0.18213782],
            ]
        ),
        atol=1e-8,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result[1],
        np.array(
            [
                [-0.00880793, -0.03672674, 0.00785002, -0.0193802, 0.02673553],
                [0.06736008, -0.01250296, -0.01666406, 0.02355922, -0.06991658],
                [0.05932112, -0.00218145, 0.01015311, -0.02424715, -0.04874478],
            ]
        ),
        atol=1e-8,
        rtol=1e-5,
    )

    # Test case 3
    result = gradient_check(
        ndl.divide_scalar, ndl.Tensor(np.random.randn(3, 5)), scalar=np.random.randn(1)
    )
    np.testing.assert_allclose(
        result,
        [
            np.array(
                [
                    [-1.48707631, -1.48707631, -1.48707631, -1.48707631, -1.48707631],
                    [-1.48707631, -1.48707631, -1.48707631, -1.48707631, -1.48707631],
                    [-1.48707631, -1.48707631, -1.48707631, -1.48707631, -1.48707631],
                ]
            )
        ],
    )

    # Test case 4
    result = gradient_check(
        ndl.matmul, ndl.Tensor(np.random.randn(1, 5)), ndl.Tensor(np.random.randn(5, 1))
    )
    np.testing.assert_allclose(
        result[0],
        np.array([[-1.63019835, 0.46278226, -0.90729836, 0.0519454, 0.72909056]]),
    )
    np.testing.assert_allclose(
        result[1],
        np.array(
            [[-0.35955316], [-0.81314628], [-1.7262826], [0.17742614], [-0.40178094]]
        ),
    )

    # Test case 5
    result = gradient_check(
        ndl.matmul, ndl.Tensor(np.random.randn(2, 4)), ndl.Tensor(np.random.randn(4, 2))
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [-1.1089845, 1.36648893, -0.04799149, 3.07466875],
                [-1.1089845, 1.36648893, -0.04799149, 3.07466875],
            ]
        ),
    )
    np.testing.assert_allclose(
        result[1],
        np.array(
            [
                [-0.55582718, -0.55582718],
                [0.26860354, 0.26860354],
                [-1.81367549, -1.81367549],
                [0.09078911, 0.09078911],
            ]
        ),
    )

    # Test case 6
    result = gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(2, 4)),
        ndl.Tensor(np.random.randn(7, 4, 2)),
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [8.64955648, 4.14059474, 2.92385998, 0.07633434],
                [8.64955648, 4.14059474, 2.92385998, 0.07633434],
            ]
        ),
    )
    np.testing.assert_allclose(
        result[1],
        np.array(
            [
                [
                    [1.04252023, 1.04252023],
                    [-0.86247764, -0.86247764],
                    [2.03109076, 2.03109076],
                    [-0.04681055, -0.04681055],
                ],
                [
                    [1.04252023, 1.04252023],
                    [-0.86247764, -0.86247764],
                    [2.03109076, 2.03109076],
                    [-0.04681055, -0.04681055],
                ],
                [
                    [1.04252023, 1.04252023],
                    [-0.86247764, -0.86247764],
                    [2.03109076, 2.03109076],
                    [-0.04681055, -0.04681055],
                ],
                [
                    [1.04252023, 1.04252023],
                    [-0.86247764, -0.86247764],
                    [2.03109076, 2.03109076],
                    [-0.04681055, -0.04681055],
                ],
                [
                    [1.04252023, 1.04252023],
                    [-0.86247764, -0.86247764],
                    [2.03109076, 2.03109076],
                    [-0.04681055, -0.04681055],
                ],
                [
                    [1.04252023, 1.04252023],
                    [-0.86247764, -0.86247764],
                    [2.03109076, 2.03109076],
                    [-0.04681055, -0.04681055],
                ],
                [
                    [1.04252023, 1.04252023],
                    [-0.86247764, -0.86247764],
                    [2.03109076, 2.03109076],
                    [-0.04681055, -0.04681055],
                ],
            ]
        ),
    )

    # Test case 7
    result = gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(3, 2, 1)),
        ndl.Tensor(np.random.randn(3, 3, 1, 2)),
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [[-2.127483], [-2.127483]],
                [[0.0838534], [0.0838534]],
                [[0.83694312], [0.83694312]],
            ]
        ),
    )
    np.testing.assert_allclose(
        result[1],
        np.array(
            [
                [
                    [[0.9685879, 0.9685879]],
                    [[-0.92489106, -0.92489106]],
                    [[0.46315764, 0.46315764]],
                ],
                [
                    [[0.9685879, 0.9685879]],
                    [[-0.92489106, -0.92489106]],
                    [[0.46315764, 0.46315764]],
                ],
                [
                    [[0.9685879, 0.9685879]],
                    [[-0.92489106, -0.92489106]],
                    [[0.46315764, 0.46315764]],
                ],
            ]
        ),
    )

    # Test case 8
    result = gradient_check(
        ndl.matmul,
        ndl.Tensor(np.random.randn(2, 4)),
        ndl.Tensor(np.random.randn(2, 4, 4, 2)),
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [-3.34425985, -0.09100445, 2.61732935, -4.45420866],
                [-3.34425985, -0.09100445, 2.61732935, -4.45420866],
            ]
        ),
    )
    np.testing.assert_allclose(
        result[1],
        np.array(
            [
                [
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                ],
                [
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                    [
                        [0.69907368, 0.69907368],
                        [-1.15740358, -1.15740358],
                        [-2.06450107, -2.06450107],
                        [-1.09915091, -1.09915091],
                    ],
                ],
            ]
        ),
    )

    # Test case 9
    result = gradient_check(
        ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(5, 4, 1)
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    # Test case 10
    result = gradient_check(
        ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(2, 2, 5)
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    # Test case 11
    result = gradient_check(ndl.negate, ndl.Tensor(np.random.randn(1, 4, 2)))
    np.testing.assert_allclose(
        result[0], np.array([[[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]])
    )

    # Test case 12
    result = gradient_check(
        ndl.transpose, ndl.Tensor(np.random.randn(3, 2, 4)), axes=(0, 2)
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ]
        ),
    )

    # Test case 13
    result = gradient_check(
        ndl.broadcast_to, ndl.Tensor(np.random.randn(7, 1)), shape=(7, 7)
    )
    np.testing.assert_allclose(
        result[0], np.array([[7.0], [7.0], [7.0], [7.0], [7.0], [7.0], [7.0]])
    )

    # Test case 14
    result = gradient_check(
        ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 5)), shape=(5, 5)
    )
    np.testing.assert_allclose(result[0], np.array([[5.0, 5.0, 5.0, 5.0, 5.0]]))

    # Test case 15
    result = gradient_check(
        ndl.broadcast_to, ndl.Tensor(np.random.randn(1)), shape=(4, 4, 4)
    )
    np.testing.assert_allclose(result[0], np.array([64.0]))

    # Test case 16
    result = gradient_check(
        ndl.broadcast_to, ndl.Tensor(np.random.randn()), shape=(1, 3, 6)
    )
    np.testing.assert_allclose(result[0], np.array([18.0]))

    # Test case 17
    result = gradient_check(
        ndl.broadcast_to, ndl.Tensor(np.random.randn(4, 4, 1)), shape=(4, 4, 6)
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [[6.0], [6.0], [6.0], [6.0]],
                [[6.0], [6.0], [6.0], [6.0]],
                [[6.0], [6.0], [6.0], [6.0]],
                [[6.0], [6.0], [6.0], [6.0]],
            ]
        ),
    )

    # Test case 18
    result = gradient_check(ndl.summation, ndl.Tensor(np.random.randn(3, 2, 1)))
    np.testing.assert_allclose(
        result[0], np.array([[[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]]])
    )

    # Test case 19
    result = gradient_check(ndl.summation, ndl.Tensor(np.random.randn(3, 6)), axes=(1,))
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    # Test case 20
    result = gradient_check(
        ndl.summation,
        ndl.Tensor(
            np.random.randn(
                7,
            )
        ),
        axes=(0,),
    )
    np.testing.assert_allclose(result[0], np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    # Test case 21
    result = gradient_check(
        ndl.summation, ndl.Tensor(np.random.randn(7, 8)), axes=(0, 1)
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )

    # Test case 22
    result = gradient_check(
        ndl.summation, ndl.Tensor(np.random.randn(5, 4, 5)), axes=(0, 1, 2)
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        ),
    )

    # Test case 23
    result = gradient_check(ndl.log, ndl.Tensor(10 + np.random.rand(5, 4)))
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [0.09920111, 0.09915411, 0.09783399, 0.09900976],
                [0.09741804, 0.09934285, 0.09934823, 0.09211262],
                [0.09840466, 0.09469982, 0.09282073, 0.09563512],
                [0.09848948, 0.0980431, 0.09584985, 0.09498269],
                [0.09662358, 0.09275165, 0.09301442, 0.09151465],
            ]
        ),
    )

    # Test case 24
    result = gradient_check(ndl.exp, ndl.Tensor(1.5 + np.random.rand(5, 4)))
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                [4.61334204, 10.9757829, 6.6363943, 10.78733214],
                [8.94222801, 12.0293411, 9.5762153, 6.45299669],
                [7.39691612, 6.52988384, 6.45536673, 5.81769712],
                [7.3593403, 8.86171174, 5.91410557, 7.57141378],
                [5.03987003, 5.25849722, 4.69644742, 11.83109746],
            ]
        ),
    )


##############################################################################
### TESTS/SUBMISSION CODE FOR find_topo_sort


def test_topo_sort():
    # Test case 1
    a1, b1 = ndl.Tensor(np.asarray([[0.88282157]])), ndl.Tensor(
        np.asarray([[0.90170084]])
    )
    c1 = 3 * a1 * a1 + 4 * b1 * a1 - a1

    soln = np.array(
        [
            np.array([[0.88282157]]),
            np.array([[2.64846471]]),
            np.array([[2.33812177]]),
            np.array([[0.90170084]]),
            np.array([[3.60680336]]),
            np.array([[3.1841638]]),
            np.array([[5.52228558]]),
            np.array([[-0.88282157]]),
            np.array([[4.63946401]]),
        ]
    )

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c1])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 2
    a1, b1 = ndl.Tensor(np.asarray([[0.20914675], [0.65264178]])), ndl.Tensor(
        np.asarray([[0.65394286, 0.08218317]])
    )
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    soln = [
        np.array([[0.65394286, 0.08218317]]),
        np.array([[0.20914675], [0.65264178]]),
        np.array([[0.19040619]]),
        np.array([[1.53101102, 0.19240724]]),
        np.array([[0.44577898]]),
        np.array([[0.63618518]]),
        np.array([[1.90855553]]),
        np.array([[3.40855553]]),
    ]

    topo_order = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

    assert len(soln) == len(topo_order)
    # step through list as entries differ in length
    for t, s in zip(topo_order, soln):
        np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    # Test case 3
    a = ndl.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = ndl.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = (a @ b + b - a) @ a

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([e])])

    soln = np.array(
        [
            np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
            np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
            np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
            np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
            np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
            np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
            np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]]),
        ]
    )

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 1
    a2, b2 = ndl.Tensor(np.asarray([[0.74683138]])), ndl.Tensor(
        np.asarray([[0.65539231]])
    )
    c2 = 9 * a2 * a2 + 15 * b2 * a2 - b2

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c2])])
    assert_allclose_list(
        topo_order,
        np.array([
            [[0.74683138]],
            [[6.72148242]],
            [[5.01981399]],
            [[0.65539231]],
            [[9.83088465]],
            [[7.34201315]],
            [[12.36182714]],
            [[-0.65539231]],
            [[11.70643483]],
        ])
    )

    # Test case 2
    a1, b1 = ndl.Tensor(np.asarray([[0.9067453], [0.18521121]])), ndl.Tensor(
        np.asarray([[0.80992494, 0.52458167]])
    )
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    topo_order2 = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]
    assert_allclose_list(
        topo_order2,
        [
            np.array([[0.80992494, 0.52458167]]),
            np.array([[0.9067453], [0.18521121]]),
            np.array([[0.83155404]]),
            np.array([[1.89619627, 1.22815061]]),
            np.array([[1.94683432]]),
            np.array([[2.77838835]]),
            np.array([[8.33516506]]),
            np.array([[9.83516506]]),
        ],
    )

    # Test case 3
    c = ndl.Tensor(np.asarray([[-0.16541387, 2.52604789], [-0.31008569, -0.4748876]]))
    d = ndl.Tensor(np.asarray([[0.55936155, -2.12630983], [0.59930618, -0.19554253]]))
    f = (c + d @ d - d) @ c

    topo_order3 = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([f])])
    assert_allclose_list(
        topo_order3,
        np.array(
            [
                [[-0.16541387, 2.52604789], [-0.31008569, -0.4748876]],
                [[0.55936155, -2.12630983], [0.59930618, -0.19554253]],
                [[-0.96142528, -0.77359196], [0.21803899, -1.23607374]],
                [[-1.12683915, 1.75245593], [-0.0920467, -1.71096134]],
                [[-0.55936155, 2.12630983], [-0.59930618, 0.19554253]],
                [[-1.6862007, 3.87876576], [-0.69135288, -1.51541881]],
                [[-0.92382877, -6.10140148], [0.58426904, -1.02673689]],
            ]
        ),
    )


##############################################################################
### TESTS/SUBMISSION CODE FOR compute_gradient_of_variables


def test_compute_gradient():
    gradient_check(
        lambda A, B, C: ndl.summation((A @ B + C) * (A @ B), axes=None),
        ndl.Tensor(np.random.randn(10, 9)),
        ndl.Tensor(np.random.randn(9, 8)),
        ndl.Tensor(np.random.randn(10, 8)),
        backward=True,
    )
    gradient_check(
        lambda A, B: ndl.summation(ndl.broadcast_to(A, shape=(10, 9)) * B, axes=None),
        ndl.Tensor(np.random.randn(10, 1)),
        ndl.Tensor(np.random.randn(10, 9)),
        backward=True,
    )
    gradient_check(
        lambda A, B, C: ndl.summation(
            ndl.reshape(A, shape=(10, 10)) @ B / 5 + C, axes=None
        ),
        ndl.Tensor(np.random.randn(100)),
        ndl.Tensor(np.random.randn(10, 5)),
        ndl.Tensor(np.random.randn(10, 5)),
        backward=True,
    )

    # check gradient of gradient
    x2 = ndl.Tensor([6])
    x3 = ndl.Tensor([0])
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1

    # Test case 1
    a = ndl.Tensor(np.array([[-0.2985143, 0.36875625], [-0.918687, 0.52262925]]))
    b = ndl.Tensor(np.array([[-1.58839928, 1.58592338], [-0.15932137, -0.55618462]]))
    c = ndl.Tensor(np.array([[-0.5096208, 0.73466865], [0.38762148, -0.41149092]]))
    d = (a + b) @ c @ (a + c)
    d.backward()
    grads = [x.grad.numpy() for x in [a, b, c]]
    assert_allclose_list(
        grads,
        [
            np.array([[1.79666176, 2.54291182], [-3.42775356, -2.6815035]]),
            np.array([[-0.45899317, 0.2872569], [-0.45899317, 0.2872569]]),
            np.array([[1.38014372, 3.50070627], [-2.401472, -3.77549271]]),
        ],
    )

    # Test case 2
    a = ndl.Tensor(
        np.array(
            [
                [0.4736625, 0.06895066, 1.36455087, -0.31170743, 0.1370395],
                [0.2283258, 0.72298311, -1.20394586, -1.95844434, -0.69535299],
                [0.18016408, 0.0266557, 0.80940201, -0.45913678, -0.05886218],
                [-0.50678819, -1.53276348, -0.27915708, -0.571393, -0.17145921],
            ]
        )
    )
    b = ndl.Tensor(
        np.array(
            [
                [0.28738358, -1.27265428, 0.32388374],
                [-0.77830395, 2.07830592, 0.99796268],
                [-0.76966429, -1.37012833, -0.16733693],
                [-0.44134101, -1.24495901, -1.62953897],
                [-0.75627713, -0.80006226, 0.03875171],
            ]
        )
    )
    c = ndl.Tensor(
        np.array(
            [
                [1.25727301, 0.39400789, 1.29139323, -0.950472],
                [-0.21250305, -0.93591609, 1.6802009, -0.39765765],
                [-0.16926597, -0.45218718, 0.38103032, -0.11321965],
            ]
        )
    )
    output = ndl.summation((a @ b) @ c @ a)
    output.backward()
    grads = [x.grad.numpy() for x in [a, b, c]]
    assert_allclose_list(
        grads,
        [
            np.array(
                [
                    [-2.21307916, 8.71540795, -8.25570267, -8.47564483, -5.36128287],
                    [-6.73758707, 4.19090003, -12.78021058, -13.00015274, -9.88579078],
                    [3.8823023, 14.8107894, -2.16032121, -2.38026337, 0.73409859],
                    [-7.11633821, 3.8121489, -13.15896172, -13.37890388, -10.26454192],
                ]
            ),
            np.array(
                [
                    [1.72156736, 1.65407439, 0.58461717],
                    [-3.27548204, -3.14706883, -1.11230213],
                    [3.16850871, 3.04428932, 1.07597567],
                    [-15.13821981, -14.54473541, -5.14070112],
                    [-3.61698879, -3.47518702, -1.22827245],
                ]
            ),
            np.array(
                [
                    [3.78576053, -6.35098929, 1.08869066, -6.68996406],
                    [3.17330266, -5.32353038, 0.912563, -5.60766603],
                    [8.0409976, -13.48957211, 2.31239111, -14.2095583],
                ]
            ),
        ],
    )


##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_loss


def test_softmax_loss_ndl():
    # test forward pass for log
    np.testing.assert_allclose(
        ndl.log(ndl.Tensor([[4.0], [4.55]])).numpy(),
        np.array([[1.38629436112], [1.515127232963]]),
    )

    # test backward pass for log
    gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5, 4)))

    X, y = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    np.random.seed(0)
    Z = ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = ndl.Tensor(y_one_hot)
    np.testing.assert_allclose(
        softmax_loss(Z, y).numpy(), 2.3025850, rtol=1e-6, atol=1e-6
    )
    Z = ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
    np.testing.assert_allclose(
        softmax_loss(Z, y).numpy(), 2.7291998, rtol=1e-6, atol=1e-6
    )

    # test softmax loss backward
    Zsmall = ndl.Tensor(np.random.randn(16, 10).astype(np.float32))
    ysmall = ndl.Tensor(y_one_hot[:16])
    gradient_check(softmax_loss, Zsmall, ysmall, tol=0.01, backward=True)

    # Test log
    np.random.seed(0)
    np.testing.assert_allclose(
        gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5, 4)))[0],
        np.array(
            [
                [0.64565553, 0.583026, 0.62392242, 0.64729813],
                [0.70241747, 0.6075725, 0.69560997, 0.52860465],
                [0.50925241, 0.72283504, 0.55812135, 0.65406719],
                [0.63773698, 0.51931956, 0.93367538, 0.91985378],
                [0.98018228, 0.54566691, 0.56238012, 0.53475588],
            ]
        ),
    )

    # Test softmax loss
    X, y = parse_mnist(
        "data/t10k-images-idx3-ubyte.gz", "data/t10k-labels-idx1-ubyte.gz"
    )

    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = ndl.Tensor(y_one_hot)
    np.testing.assert_allclose(
        softmax_loss(
            ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32)), y
        ).numpy(),
        2.3025851249694824,
    )
    np.random.seed(0)
    np.testing.assert_allclose(
        softmax_loss(
            ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32)), y
        ).numpy(),
        2.732871673844163,
    )


##############################################################################
### TESTS/SUBMISSION CODE FOR nn_epoch


def test_nn_epoch_ndl():
    # test forward/backward pass for relu
    np.testing.assert_allclose(
        ndl.relu(
            ndl.Tensor(
                [
                    [-46.9, -48.8, -45.45, -49.0],
                    [-49.75, -48.75, -45.8, -49.25],
                    [-45.65, -45.25, -49.3, -47.65],
                ]
            )
        ).numpy(),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    gradient_check(ndl.relu, ndl.Tensor(np.random.randn(5, 4)))

    # test nn gradients
    np.random.seed(0)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    W1_0, W2_0 = W1.copy(), W2.copy()
    W1 = ndl.Tensor(W1)
    W2 = ndl.Tensor(W2)
    X_ = ndl.Tensor(X)
    y_one_hot = np.zeros((y.shape[0], 3))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    dW1 = nd.Gradient(
        lambda W1_: softmax_loss(
            ndl.relu(X_ @ ndl.Tensor(W1_).reshape((5, 10))) @ W2, y_
        ).numpy()
    )(W1.numpy())
    dW2 = nd.Gradient(
        lambda W2_: softmax_loss(
            ndl.relu(X_ @ W1) @ ndl.Tensor(W2_).reshape((10, 3)), y_
        ).numpy()
    )(W2.numpy())
    W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
    np.testing.assert_allclose(
        dW1.reshape(5, 10), W1_0 - W1.numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        dW2.reshape(10, 3), W2_0 - W2.numpy(), rtol=1e-4, atol=1e-4
    )

    # test full epoch
    X, y = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    np.random.seed(0)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
    np.testing.assert_allclose(
        np.linalg.norm(W1.numpy()), 28.437788, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(W2.numpy()), 10.455095, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        loss_err(ndl.relu(ndl.Tensor(X) @ W1) @ W2, y),
        (0.19770025, 0.06006667),
        rtol=1e-4,
        atol=1e-4,
    )

def test_nn_epoch_ndl_2():
    X, y = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )

    # First test case
    np.random.seed(1)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X[:100], y[:100], W1, W2, lr=0.1, batch=100)

    np.testing.assert_allclose(np.linalg.norm(W1.numpy()), 27.988908036076715)
    np.testing.assert_allclose(np.linalg.norm(W2.numpy()), 9.796814125323062)

    # Second test case
    np.random.seed(1)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)

    np.testing.assert_allclose(np.linalg.norm(W1.numpy()), 28.528667964000803)
    np.testing.assert_allclose(np.linalg.norm(W2.numpy()), 10.59444107121055)
    np.testing.assert_allclose(
        loss_err(ndl.Tensor(np.maximum(X @ W1.numpy(), 0)) @ W2, y),
        (0.19377939086750373, 0.06016666666666667),
    )

