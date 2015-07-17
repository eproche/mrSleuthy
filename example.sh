#!/bin/bash
echo '
   _____         _________.__                 __  .__           
  /     \_______/   _____/|  |   ____  __ ___/  |_|  |__ ___.__.
 /  \ /  \_  __ \_____  \ |  | _/ __ \|  |  \   __\  |  <   |  |
/    Y    \  | \/        \|  |_\  ___/|  |  /|  | |   Y  \___  |
\____|__  /__| /_______  /|____/\___  >____/ |__| |___|  / ____|
        \/             \/           \/                 \/\/     
 __      __       .__                                       _____.___.             
/  \    /  \ ____ |  |   ____  ____   _____   ____   ______ \__  |   | ____  __ __ 
\   \/\/   // __ \|  | _/ ___\/  _ \ /     \_/ __ \ /  ___/  /   |   |/  _ \|  |  \
 \        /\  ___/|  |_\  \__(  <_> )  Y Y  \  ___/ \___ \   \____   (  <_> )  |  /
  \__/\  /  \___  >____/\___  >____/|__|_|  /\___  >____  >  / ______|\____/|____/ 
       \/       \/          \/            \/     \/     \/   \/                    
'
sleep 3
echo "enter sleuth to see available commands"
sleep 1
echo
sleuth
sleep 10
echo
echo "sleuth readin --help"
sleep 1
echo
sleuth readin --help
sleep 5
echo
echo 'sleuth word2vec --help'
sleep 1
echo
sleuth word2vec --help
sleep 2
echo
echo 'sleuth output --help'
sleep 1
echo
sleuth output --help
sleep 5
echo
echo 'Now we will run through an example'
sleep 1
echo
echo "First, we will read and preprocess study.csv"
sleep 1
echo
echo "sleuth readin 'csv/study.csv'"
sleuth readin 'csv/study.csv'
echo
sleep 1
echo "Now we will generate a tf-idf similarity matrix"
sleep 1
echo 
echo 'sleuth tfidf'
sleuth tfidf
echo
sleep 1
echo "Let's generate some visual outputs"
sleep 1
echo
echo 'sleuth output --mds --spring --thresh1=0.07 --sep=200'
echo
echo 'close figure displays to continue'
sleuth output --mds --spring --thresh1=0.07 --sep=200
echo 
echo "let's do some exploring!"
sleep 1
echo
echo 'sleuth output --explore --thresh1=0.03 --thresh2=0.25 --step=0.02 --sep=200'
sleuth output --explore --thresh1=0.03 --thresh2=0.25 --step=0.02 --sep=200
sleep 1
echo
echo 'we can also generate identity matrices and confusion matrices'
echo
sleep 1
echo 'sleuth output --con --sep=200 --no_thumb'
echo
sleuth output --con --no_thumb --sep=200
echo "figure saved in current working directory as con_mat.png"
echo
echo "so all of these results were based on the raw tf-idf matrix"
sleep 1.5
echo
echo "The full power of mrSleuthy comes from word2vec and the google news corpus"
echo
sleep 2
echo "You'll have to download this yourself to run word2vec command on new inputs"
echo
sleep 2
echo "However, we've saved the results of study.csv in a pickle, so running word2vec will grab from the pickle jar"
echo
sleep 2
echo "sleuth word2vec"
sleuth word2vec
echo
sleep 1
echo "sleuth output --mds --spring --thresh1=0.5 --sep=200"
sleuth output --mds --spring --thresh1=0.5 --sep=200
sleep 1
echo
echo "wow cool"
echo
sleep 1
echo "let's take a look at that sick nasty clustered identity matrix. (saved in current working directory for your perusing pleasure)"
echo 
sleep 1
echo 'sleuth output --iden --no_thumb'
echo
sleuth output --iden --no_thumb
echo
sleep 1
echo "now it's your turn!"
sleep 1

COUNT=100
for i in `seq 1 10`;
do 
	banner -w $COUNT 0
	sleep 1.5
	let COUNT-=10
done
echo "goodbye, friend"
sleep 1
