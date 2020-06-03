# Kissing Number Problem in 3 Dimensions

This is a study program for validating kissing problem in 3 dimensions based on an extension of Delsarte's method.

## References

- Musin, O. [_The Kissing Problem in Three Dimensions_](https://arxiv.org/pdf/math/0410324.pdf). Discrete Comput Geom 35, 375–384 (2006).

## Prerequisite

This project uses

- [NumPy](https://numpy.org/),
- [SciPy](https://www.scipy.org/),
- [Matplotlib](https://matplotlib.org/), and
- [SymPy](https://www.sympy.org/en/,index.html).

You can install them using `pip install -r requirements.txt`.

## Usage

You can execute `examples/s2_verify.py` via commands

```bash
cd examples
python3 s2_verify.py
```

Possible result is shown as follows:

```txt
h2: 12.874869834882462
h4: 12.917070958008141
    12.918143697010624
h3: 12.942527302266296
    12.964797422562022
    12.9508335817848
    12.960647320409375
    12.951894003307714
```

## Navigation

In this project, you can see

- `lib.py` for core features, and
- `util.py` for packed tools.
