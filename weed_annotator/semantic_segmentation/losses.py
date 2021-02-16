import segmentation_models_pytorch as smp
from weed_annotator.semantic_segmentation.lovasz_losses import lovasz_softmax, lovasz_hinge

class LovaszLoss(smp.utils.base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = smp.utils.base.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        loss = lovasz_hinge(y_pr, y_gt, per_image=False, ignore=self.ignore_channels) # no activation function here
        # loss = lovasz_softmax(y_pr, y_gt, per_image=False, classes=[1], ignore=self.ignore_channels)
        return loss

class WeightedDiceLoss(smp.utils.base.Loss):
    def __init__(self, eps=1., activation=None, ignore_channels=None, weight=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = smp.utils.base.Activation(activation)
        self.dice_simple = smp.utils.losses.DiceLoss()
        self._weight = weight

    def forward(self, y_pr, y_gt, weight=None):
        y_pr = self.activation(y_pr)
        loss = 0
        if weight != None:
            self._weight = weight
        if self._weight != None:
            for i in range(y_pr.size(1)):
                loss += self.dice_simple(y_pr[:, i, :, :], y_gt[:, i, :, :]) * self._weight[i]
            loss = loss/len(self._weight)
        else:
            loss += self.dice_simple(y_pr, y_gt)
        return loss
