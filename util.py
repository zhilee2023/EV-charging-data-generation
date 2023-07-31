

## make the start hour begin at 6:00 AM
def reset_startT_6(x):
    x=x-6
    if x<0:
        x=x+24
    return x/24
