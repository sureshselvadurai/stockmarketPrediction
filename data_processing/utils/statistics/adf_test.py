from statsmodels.tsa.stattools import adfuller


def validate(data):
    # Perform Augmented Dickey-Fuller utils
    result = adfuller(data['Close'])

    # Extracting ADF Statistic, p-value, and Critical Values
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Print ADF Test Results
    print("\nADF Test Results:")
    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
    print(f'Critical Values: {critical_values}')

    # Check for stationary based on p-value and ADF Statistic
    if p_value < 0.05:  # Common significance level is 0.05
        print("\nResult: Reject the null hypothesis")
        print("Conclusion: The time series is likely stationary.")
    else:
        print("\nResult: Fail to reject the null hypothesis")
        print("Conclusion: The time series is likely non-stationary.")

    # Additional check using ADF Statistic and Critical Values
    if adf_statistic < critical_values['1%']:
        print("\nAdditional Check: ADF Statistic is more negative than the 1% Critical Value")
        print("Conclusion: Strong evidence to reject the null hypothesis of non-stationarity.")
    elif adf_statistic < critical_values['5%']:
        print("\nAdditional Check: ADF Statistic is more negative than the 5% Critical Value")
        print("Conclusion: Moderate evidence to reject the null hypothesis of non-stationarity.")
    else:
        print("\nAdditional Check: ADF Statistic is more negative than the 10% Critical Value")
        print("Conclusion: Weak evidence to reject the null hypothesis of non-stationarity.")


