import os
from args import args
from train import TripleAD


if __name__ == '__main__':

    if not os.path.exists('.checkpoints'):
        os.makedirs('.checkpoints')

    train = TripleAD(args)

    # pre-train Attribute model
    for epoch in range(args.att_epoch):
        train.pre_train_att()
    # train.save_checkpoint(ts='att')
    print('[AttModule Pre-training] complete!')

    # pre-train Structure model
    for epoch in range(args.str_epoch):
        train.pre_train_str()
    # train.save_checkpoint(ts='str')
    print('[StrModule Pre-training] complete!')

    # load pre-train models
    # train.load_checkpoint(ts='att')
    # train.load_checkpoint(ts='str')

    # train attribute model
    for epoch in range(args.att_epoch):
        train.train_attModule(epoch)
    # train.save_checkpoint(ts='att')
    print("[AttModule Training] complete!")

    # train structure model
    for epoch in range(args.str_epoch):
        train.train_strModule(epoch)
    # train.save_checkpoint(ts='str')
    print("[StrModule Training] complete!")

    # for epoch in range(args.mix_epoch):
    #     train.train_attModule(epoch)
    # # train.save_checkpoint(ts='mix')
    # print("[MixModule Training] complete!")

    res = train.test()

    print('[FINAL RESULT] AUC_ROC: {:.2f}'.format(res))



