for i in $(seq 1 4);
do
    echo "Freddie Mercury $i"
    python cktinderator.py --image target/faces/real_${i}_freddie.png --dna target/freddie_mercury.txt  --output results/6-mercury-enface-aligned/2024-06-09/params/freddie-${i}-lr-low.txt --generated results/6-mercury-enface-aligned/2024-06-09/generated/freddie-${i}-lr-low.pdf
done