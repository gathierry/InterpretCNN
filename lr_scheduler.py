
class LRScheduler():
    def __init__(self, base_lr, epochs, patience, factor, min_lr, early_stop, best_loss=float('inf')):
        self.patience = patience
        self.base_lr = base_lr
        self.epochs = epochs  # list
        self.factor = factor
        self.min_lr = min_lr
        self.early_stop = early_stop
        self.current_lr = base_lr
        self.best_loss = best_loss
        self.tolerence = 0

    def update_by_rule(self, current_loss):
        if current_loss <= self.best_loss:
            self.best_loss = current_loss
            self.tolerence = 0
        else:
            self.tolerence += 1
            if self.tolerence >= self.patience:
                tmp_lr = self.current_lr * self.factor
                if tmp_lr >= self.min_lr:
                    self.current_lr = tmp_lr
                    self.tolerence = 0
                else:
                    self.early_stop -= 1
                    if self.early_stop <= 0:
                        return None
        return self.current_lr

    def update_by_iter(self, current_epoch):
        p = 0
        for k, e in enumerate(self.epochs):
            if current_epoch >= e:
                p = k + 1
        self.current_lr = self.base_lr * (self.factor**p)
        return self.current_lr


if __name__ == '__main__':
    lrs = LRScheduler(1e-2, [3, 6], 3, 0.1, 1e-5, 5)
    for e in range(20):
        # a = float(input())
        # b = lrs.update_by_rule(a)
        b = lrs.update_by_iter(e)
        print(b)