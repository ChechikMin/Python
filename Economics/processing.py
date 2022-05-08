import pandas as pd
import numpy as np


def get_coef_annuet(month_percent, month_credit):
    # formulae from https://journal.tinkoff.ru/guide/credit-payment/
    coef_annuetniet = month_percent * (1 + month_percent) ** month_credit / ((1 + month_percent) ** month_credit - 1)
    return coef_annuetniet


def calculate_param_of_credit(sum_expceted, age, residual, month_expected=12):
    age_threshold = 80  # this threshold adjusts max age of credit repayments, [years of person]
    perc_credit = [0.2, 0.15, 0.12, 0.1, 0.08, 0.06]  # may be changed

    # initial iteration
    sum_credit = sum_expceted
    month_credit = month_expected

    # main loop
    while month_credit/12 + age < age_threshold:

        for percent in perc_credit:

            # calculate payment_per_month
            coef_annuet = get_coef_annuet(percent / 12, month_credit)
            payment_per_month_credit = sum_credit * coef_annuet

            # checked the condition
            if (payment_per_month_credit < residual) and (month_credit/12 + age < age_threshold):
                return [sum_credit, payment_per_month_credit, percent, month_credit]

        # years are varied
        month_credit += 1

    print('Невозможно выдать кредит на запрошенную сумму')
    return [0, 0, 0, 0]
