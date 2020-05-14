#!/bin/bash
chmod 600 tunnel.openssh
ssh -i tunnel.openssh -L 8122:172.16.82.107:22 -L 8888:172.16.82.107:8888 -L 7180:172.16.82.107:7180 -N studproxy@class.pmoproject.ru
