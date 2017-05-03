for i in `seq 100000 10000 200000`
do
    python ./train.py --train_size=$i --out="result$i" --gpu=1
done
