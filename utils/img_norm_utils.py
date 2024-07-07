def denormit(x):
    out=(x+1.0)/2.0
    return out.clamp_(0,1.0)

def normit(x):
    out=2*x-1.0
    return out.clamp_(-1.0,1.0)