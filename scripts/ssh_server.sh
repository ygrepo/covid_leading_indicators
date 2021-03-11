#!/usr/bin/expect

#SERVERS="odinrnd.com"
#for host in $(cat SERVERS);
#do
set host "odinrnd.com"
spawn ssh -p 1988 $host
expect "gryv9001@odinrnd.com's password: "
#puts "Hi"
send -- "plate-perspective-calling-option"
# send "command1\n"
# send "command2\n"
#done

