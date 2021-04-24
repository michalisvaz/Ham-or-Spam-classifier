from math import log2

# return (ig, p_x1_ham, p_x0_ham)
def calculate_ig(x1, x1_ham, x1_spam, total_ham, total_spam):
    total_mails = total_ham + total_spam
    if x1 == 0 or x1 == total_mails:
        return (0, total_ham/total_mails, total_ham/total_mails)
    x0 = total_mails - x1
    x0_spam = total_spam - x1_spam
    x0_ham = total_ham - x1_ham
    P_ham = total_ham/total_mails
    P_spam = 1 - P_ham
    # we do not check P_ham > 0 and P_spam > 0 
    # because if we had only spam or only ham data, the exercise would be pointless
    h = - P_ham * log2(P_ham) - P_spam * log2(P_spam) 
    P_x0_spam = x0_spam / x0
    P_x0_ham = x0_ham / x0
    P_x1_spam = x1_spam / x1
    P_x1_ham = x1_ham / x1
    # This is to avoid trying to calculate log(0)
    # We use the fact that lim(xlogx)=0 as x-->0
    # The base of the logarithm makes no difference in the above limit
    # Also note that it is impossible P_x0_ham = P_x0_spam = 0 (both probabilities being 0)
    # If we didn't do the following (and the above) we had many cases with ig<0
    # Now if we use all the pu3 data just to test if we calculate the ig correctly 
    # (we won't use all the data as training data simultaneously, this was only done to test this function with more data)
    # there is only one example with ig<0. The following:
    # 1826,-1.8735013540549517e-16
    # which is probably due to numerical errors (you can notice that it is very close to 0)
    # To correct these numerical we added an if ig<0 at the end of the function
    if P_x0_ham == 0:
        h0 = - P_x0_spam * log2(P_x0_spam)
    elif P_x0_spam == 0:
        h0 = - P_x0_ham * log2(P_x0_ham)
    else:
        h0 = - P_x0_ham * log2(P_x0_ham) - P_x0_spam * log2(P_x0_spam)
    # Same as above
    if P_x1_ham == 0:
        h1 = - P_x1_spam * log2(P_x1_spam)
    elif P_x1_spam == 0:
        h1 = - P_x1_ham * log2(P_x1_ham)
    else:
        h1 = - P_x1_ham * log2(P_x1_ham) - P_x1_spam * log2(P_x1_spam)
    ig = h - h0 * (x0/total_mails) - h1 * (x1/total_mails)
    if ig < 0:
        return (0, P_x1_ham, P_x0_ham)
    else:
        return (ig, P_x1_ham, P_x0_ham)
