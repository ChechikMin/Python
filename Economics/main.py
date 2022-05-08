import pandas as pd
from processing import full_column_residual, find_emission
import processing as proc


if __name__ == '__main__':

    # calculate sum of credit
    #input from user
    test_age = 40
    expected_sum = 1e6
    expected_month = 25
    month_residual = 10000
    #input from user

    delta_for_sum = expected_sum * 0.05  # vary the sum of credit with step 5% of expected sum
    sum_of_credit = [expected_sum - i*delta_for_sum for i in range(10)]

    Bank_helper_result = pd.DataFrame({'Sum of credit': [''], 'payment per month': [''], 'bank percent': [''], 'month credit': [''], 'overpayment': ['']})
    for dif_sum in sum_of_credit:
        current_result = proc.calculate_param_of_credit(dif_sum, test_age, month_residual, expected_month)
        if current_result[0]:
            Bank_helper_result.loc[len(Bank_helper_result.index)] = [current_result[0], current_result[1], current_result[2], current_result[3], abs(current_result[0] - current_result[1]*current_result[3])]

    print(Bank_helper_result)








