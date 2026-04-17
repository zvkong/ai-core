from src.z2h.micrograd_pytorch.value import Value

def numerical_grad_1d(fn, x, h=1e-6):
    out = (fn(x + h) - fn(x - h))/(2*h)
    return out

def numerical_grad_2d_x(fn, x, y, h=1e-6):
    out_x = (fn(x + h, y) - fn(x - h, y))/(2*h)
    return out_x

def numerical_grad_2d_y(fn, x, y, h=1e-6):
    out_y = (fn(x, y + h) - fn(x, y - h))/(2*h)
    return out_y

def test_backward_populates_grad():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a
    c.backward()

    assert c.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0

def test_grad_quadratic():
    x = Value(2.0)
    y = x**2 + 2*x + 1
    y.backward()

    def fn(v):
        return v**2 + 2*v + 1

    num_grad = numerical_grad_1d(fn, 2.0)

    assert abs(x.grad - num_grad) < 1e-5


def test_grad_bilinear():
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x + y
    z.backward()

    def fn(a, b):
        return a * b + a + b

    num_grad_x = numerical_grad_2d_x(fn, 2.0, 3.0)
    num_grad_y = numerical_grad_2d_y(fn, 2.0, 3.0)

    assert abs(x.grad - num_grad_x) < 1e-5
    assert abs(y.grad - num_grad_y) < 1e-5

def test_grad_tanh_composite():
    x = Value(0.5)
    y = (x**2 + 3*x).tanh()
    y.backward()

    def fn(v):
        return math.tanh(v**2 + 3*v)

    num_grad = numerical_grad_1d(fn, 0.5)

    assert abs(x.grad - num_grad) < 1e-5 