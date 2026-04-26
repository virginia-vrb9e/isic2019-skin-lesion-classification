default:
	@cat makefile

run:
	mkdir -p logs
	RUN_ID=$$(date +%Y%m%d_%H%M%S); \
	PYTHONUNBUFFERED=1 python 4_25_EFN_HS_Ablation1.py --wandb 2>&1 | tee -a "logs/train_$$RUN_ID.log"
