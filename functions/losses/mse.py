##################################################
## {loss function MSE}
##################################################
## {License_info}
##################################################
## Author: {Sahar Yousefi}
## Copyright: Copyright {2020}, {LUMC}
## Credits: [Sahar Yousefi]
## License: {GPL}
## Version: 1.0.0
## Mmaintainer: {Sahar Yousefi}
## Email: {s.yousefi.radi[at]lumc.nl}
## Status: {Research}
##################################################
import tensorflow as tf
def mean_squared_error( labels, logit):
    loss = tf.losses.mean_squared_error(
        labels=labels,
        predictions=logit)
    return loss