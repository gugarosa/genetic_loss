# Common variables
DATA="mnist"
OUTPUT="genetic_loss.pkl"
BATCH_SIZE=128
EPOCHS=5
DEVICE="cpu"

# Architecture variables
MODEL="mlp"
N_INPUT=784
N_HIDDEN=128
N_CLASSES=10
LR=0.0001

# Optimization variables
N_AGENTS=1
N_ITER=1
MIN_DEPTH=1
MAX_DEPTH=5
INIT_LOSS_PROB=1.0
N_RUNS=1

# Creating a loop
for i in $(seq 1 $N_RUNS); do
    # Finding the optimized loss
    python find_optimized_loss.py $DATA $MODEL $OUTPUT -batch_size $BATCH_SIZE -n_input $N_INPUT -n_hidden $N_HIDDEN -n_classes $N_CLASSES -lr $LR -epochs $EPOCHS -n_agents $N_AGENTS -n_iter $N_ITER -min_depth $MIN_DEPTH -max_depth $MAX_DEPTH -init_loss_prob $INIT_LOSS_PROB -device $DEVICE -seed $i --shuffle

    # Running the final evaluation
    python evaluate_optimized_loss.py $DATA $MODEL $OUTPUT -batch_size $BATCH_SIZE -n_input $N_INPUT -n_hidden $N_HIDDEN -n_classes $N_CLASSES -lr $LR -epochs $EPOCHS -device $DEVICE -seed $i --shuffle
done