# Basic Financial Calculations with Python
# Calculate present value

# Import libraries
import numpy_financial as npf

present_value = npf.pv(rate=0.03, nper=5, pmt=0, fv=1000)
print("\nPresent Value:", round(present_value, 2))

# Calculate future value
future_value = npf.fv(rate=0.03, nper=5, pmt=0, pv=-1000)
print("Future Value:", round(future_value, 2))
