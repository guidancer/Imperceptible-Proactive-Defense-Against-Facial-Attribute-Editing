import math
def magnitude(x):
    return math.floor(math.log10(x))
def efficient_loss_balance(loss_base, loss_to_adjust):
    return loss_base/loss_to_adjust

def normal_loss_balance(loss_base, loss_to_adjust):
    mag_base = magnitude(loss_base)
    mag_adjust = magnitude(loss_to_adjust)
    mag_differ =  mag_base-mag_adjust
    return math.pow(10,mag_differ)