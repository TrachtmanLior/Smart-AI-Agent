#!bin/bash
echo 'started running BruteForce-Web'
python main.py ../data/cic2018/Thurs-22-02-BruteForce-Web-benign.csv 4000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-BruteForce-Web_4000_1.txt
python main.py ../data/cic2018/Thurs-22-02-BruteForce-Web-benign.csv 2000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-BruteForce-Web_2000_1.txt
python main.py ../data/cic2018/Thurs-22-02-BruteForce-Web-benign.csv 1000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-BruteForce-Web_1000_1.txt

echo 'started running BruteForce-XSS'
python main.py ../data/cic2018/Thurs-22-02-BruteForce-XSS-benign.csv 4000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-BruteForce-XSS_4000_1.txt
python main.py ../data/cic2018/Thurs-22-02-BruteForce-XSS-benign.csv 2000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-BruteForce-XSS_2000_1.txt
python main.py ../data/cic2018/Thurs-22-02-BruteForce-XSS-benign.csv 1000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-BruteForce-XSS_1000_1.txt

echo 'started running DoS-SlowHTTPTest'
python main.py ../data/cic2018/Fri-16-02-DoS-SlowHTTPTest-benign.csv 4000 172.31.69.25 13.59.126.31 > ../output/cic2018/Fri-16-02-DoS-SlowHTTPTest_4000_1.txt
python main.py ../data/cic2018/Fri-16-02-DoS-SlowHTTPTest-benign.csv 2000 172.31.69.25 13.59.126.31 > ../output/cic2018/Fri-16-02-DoS-SlowHTTPTest_2000_1.txt
python main.py ../data/cic2018/Fri-16-02-DoS-SlowHTTPTest-benign.csv 1000 172.31.69.25 13.59.126.31 > ../output/cic2018/Fri-16-02-DoS-SlowHTTPTest_1000_1.txt

echo 'started running Infiltration1'
python main.py ../data/cic2018/Wed-28-02-Infiltration1-benign.csv 4000 172.31.69.24 13.58.225.34 > ../output/cic2018/Wed-28-02-Infiltration1_4000_1.txt
python main.py ../data/cic2018/Wed-28-02-Infiltration1-benign.csv 2000 172.31.69.24 13.58.225.34 > ../output/cic2018/Wed-28-02-Infiltration1_2000_1.txt
python main.py ../data/cic2018/Wed-28-02-Infiltration1-benign.csv 1000 172.31.69.24 13.58.225.34 > ../output/cic2018/Wed-28-02-Infiltration1_1000_1.txt

echo 'started running Infiltration2'
python main.py ../data/cic2018/Wed-28-02-Infiltration2-benign.csv 4000 172.31.69.24 13.58.225.34 > ../output/cic2018/Wed-28-02-Infiltration2_4000_1.txt
python main.py ../data/cic2018/Wed-28-02-Infiltration2-benign.csv 2000 172.31.69.24 13.58.225.34 > ../output/cic2018/Wed-28-02-Infiltration2_2000_1.txt
python main.py ../data/cic2018/Wed-28-02-Infiltration2-benign.csv 1000 172.31.69.24 13.58.225.34 > ../output/cic2018/Wed-28-02-Infiltration2_1000_1.txt

echo 'started running SQL-Injection'
python main.py ../data/cic2018/Thurs-22-02-SQL-Injection-benign.csv 4000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-SQL-Injection_4000_1.txt
python main.py ../data/cic2018/Thurs-22-02-SQL-Injection-benign.csv 2000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-SQL-Injection_2000_1.txt
python main.py ../data/cic2018/Thurs-22-02-SQL-Injection-benign.csv 1000 172.31.69.28 18.218.115.60 > ../output/cic2018/Thurs-22-02-SQL-Injection_1000_1.txt

echo 'started running SSH-Bruteforce'
python main.py ../data/cic2018/Wed-14-02-SSH-Bruteforce-benign.csv 4000 172.31.69.25 13.58.98.64 > ../output/cic2018/Wed-14-02-SSH-Bruteforce_4000_1.txt
python main.py ../data/cic2018/Wed-14-02-SSH-Bruteforce-benign.csv 2000 172.31.69.25 13.58.98.64 > ../output/cic2018/Wed-14-02-SSH-Bruteforce_2000_1.txt
python main.py ../data/cic2018/Wed-14-02-SSH-Bruteforce-benign.csv 1000 172.31.69.25 13.58.98.64 > ../output/cic2018/Wed-14-02-SSH-Bruteforce_1000_1.txt

echo 'started running FTP-BruteForce'
python main.py ../data/cic2018/Wed-14-02-FTP-BruteForce.csv 4000 172.31.69.25 18.221.219.4 > ../output/cic2018/Wed-14-02-FTP-BruteForce_4000_1.txt
python main.py ../data/cic2018/Wed-14-02-FTP-BruteForce.csv 2000 172.31.69.25 18.221.219.4 > ../output/cic2018/Wed-14-02-FTP-BruteForce_2000_1.txt
python main.py ../data/cic2018/Wed-14-02-FTP-BruteForce.csv 1000 172.31.69.25 18.221.219.4 > ../output/cic2018/Wed-14-02-FTP-BruteForce_1000_1.txt

echo 'started running DoS-GoldenEye'
python main.py ../data/cic2018/Thurs-15-02-DoS-GoldenEye-benign.csv 4000 172.31.69.25 18.219.211.138 > ../output/cic2018/Thurs-15-02-DoS-GoldenEye_4000_1.txt
python main.py ../data/cic2018/Thurs-15-02-DoS-GoldenEye-benign.csv 2000 172.31.69.25 18.219.211.138 > ../output/cic2018/Thurs-15-02-DoS-GoldenEye_2000_1.txt
python main.py ../data/cic2018/Thurs-15-02-DoS-GoldenEye-benign.csv 1000 172.31.69.25 18.219.211.138 > ../output/cic2018/Thurs-15-02-DoS-GoldenEye_1000_1.txt

echo 'started running DoS-Slowloris'
python main.py ../data/cic2018/Thurs-15-02-DoS-Slowloris-benign.csv 4000 172.31.69.25 18.217.165.70 > ../output/cic2018/Thurs-15-02-DoS-Slowloris_4000_1.txt
python main.py ../data/cic2018/Thurs-15-02-DoS-Slowloris-benign.csv 2000 172.31.69.25 18.217.165.70 > ../output/cic2018/Thurs-15-02-DoS-Slowloris_2000_1.txt
python main.py ../data/cic2018/Thurs-15-02-DoS-Slowloris-benign.csv 1000 172.31.69.25 18.217.165.70 > ../output/cic2018/Thurs-15-02-DoS-Slowloris_1000_1.txt


#main.py input_file_path 4000
#pwd: /Users/liortrachtman/Ran_Final_Project/Smart-AI-Agent/Agent/data/
#command: cd Agent
# python GNN_Anomaly_Detection-master/src/main.py data/2017/SQL_injection.csv 4000 > ../data/2017/SQL_injection_4000.txt

