# projlib.io

Utilities for different input/output formats.

## Examples

In the example below we load a `.data` file using the `load_data` function from the `vicg` module. The output format is a pandas `DataFrame`. Notice that the class data (`cdata`), which is usually in the last column of the file, is returned separately from the actual data.

```python
from projlib.io.vicg import load_data

df, cdata = load_data("color.data")
df.head()
```

Output:

```
         1         2         3         4   ...        29        30        31        32
0                                          ...                                        
1  0.002188  0.000000  0.000000  0.620521  ...  0.000417  0.000000  0.000000  0.000000
2  0.002917  0.315417  0.188854  0.004440  ...  0.007917  0.326562  0.133958  0.011771
3  0.000313  0.009825  0.008978  0.663125  ...  0.000834  0.001701  0.004701  0.288647
4  0.111667  0.123855  0.078230  0.085486  ...  0.117604  0.010209  0.000834  0.000030
5  0.329803  0.522930  0.034487  0.011571  ...  0.015950  0.040522  0.006979  0.000417
```

Loading a projection with `load_data` is basically the same:

```python
from projlib.io.vicg import load_proj

df, cdata = load_proj("projections/color-ipca.data")
df.head()
```

Output:

```
          x         y
0                    
1  0.297186  0.107123
2  0.218629  0.069451
3  0.288687  0.111987
4  0.108259  0.006495
5  0.242876  0.041998
```