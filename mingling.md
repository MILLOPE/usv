## test 
python test.py --env USV -n 16 --u-ref --epi 1 --area-size 4 --obs 0
python test.py --env USV -n 16 --u-ref --epi 1 --area-size 4 --obs 0 --seed 123 --max-step 500
python test.py --env DubinsCar -n 16 --u-ref --epi 1 --area-size 4 --obs 0

## delete
ls /root/gcbfplus/logs/USV/nominal/videos/0331*.mp4
rm /root/gcbfplus/logs/USV/nominal/videos/0331*.mp4