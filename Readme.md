# dgl 0.5版本 graphsage

## 运行

在`graph.json`中加上数据集, 如movie

```
python dataprocess.py # 数据处理，得到目标文件
```
```
python main.py --device cuda:0 --log movie.txt --weight_decay 0  --batch_size 200000 --dataset movie --out_agg mean
```

## 参数
```bash
  --path PATH           dataset path
  --dataset DATASET     dataset name
  --num_layers NUM_LAYERS
                        number layers
  --in_feats IN_FEATS   dim of input features
  --n_hidden N_HIDDEN   dim of hidden features
  --out_feats OUT_FEATS
                        dim of output
  --activation ACTIVATION
                        activation function, i.e. [relu, leak_relu]
  --device DEVICE       device, i.e. [cpu, cuda:0, ...]
  --num_epochs NUM_EPOCHS
                        number of epochs
  --num_negs NUM_NEGS   number of negative samples
  --fanouts FANOUTS     sample number for each node in each layer, e.g. 10,25
  --batch_size BATCH_SIZE
                        batch size, can be larger
  --lr LR               learning rate
  --agg AGG             model aggregation type, i.e. [mean, lightgcn, weight,
                        gcn, neg]
  --weight_decay WEIGHT_DECAY
                        weight for L2 loss
  --step_size STEP_SIZE
                        step size for learning rate decrease
  --gamma GAMMA         Reduction ratio for each step
  --dropout DROPOUT     drop out ratio
  --num_workers NUM_WORKERS
                        number of sampler workers, now we must set it to 0,
                        because of bugs
  --out_agg OUT_AGG     the output aggregation type, i.e. [concat, mean, others]
  --log LOG             log file
  --full_sample         whether full sample nodes
  --has_neg             whether have neg file, mainly used by hhy
  --has_kg              whether have kg
```
## 输入文件格式
```
user_pos.dat 如果二部图，只需要这一个文件
user_neg.dat 一般不需要这个，has_neg=True
格式：
head \t tail \t weight \n

kg.dat 如果有kg的话，即has_kg=True
格式：
head \t relation \t tail \t weight \n
其中head和tail可以当user和item，weight如果没有可以设置成1

```

