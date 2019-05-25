# pytorch_vision_learning
一个从alexnet开始的视觉任务源码

文件介绍
imagenet/main.py（以下简称main.py）
从pytorch的官方github下载 pytorch/examples里面的imagenet下面拷贝过来的

- impl_alex_to_hymenoptera.py 也是使用了main.py里面的几个函数
``` 
if __name__ == '__main__':
    args = imagenet_util.parser.parse_args(sys.argv)
    print(args)
    # train_and_valide_on_hymenoptera(args)
    test_best_model_on_hymenoptera(args)
    
    
#test_best_model_on_hymenoptera 功能是加载模型进行test集识别，并展示图片
#，图片的title由ground_truth值以及preds值组成
#train_and_valide_on_hymenoptera 功能是训练模型
```

- impl_alex_to_cifar.py 这个使用了main.py里面的几个函数,用法和上面的一致，写法也一致
