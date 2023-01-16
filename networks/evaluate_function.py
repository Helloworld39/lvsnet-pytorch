import torch


def dice_score(predict: torch.Tensor, target: torch.Tensor, threshold=0.5):
    batch_size = predict.size()[0]
    predict = predict.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # 二值化
    predict[predict < threshold] = 0.0
    predict[predict >= threshold] = 1.0

    intersection = (predict * target).sum(1)
    t1, t2 = predict.sum(1), target.sum(1)
    score = (2 * intersection + 1e-5) / (t1 + t2 + 1e-5)

    return score.sum(0).item() / batch_size
