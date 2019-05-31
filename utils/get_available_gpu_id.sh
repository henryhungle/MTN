#!/bin/bash
nvidia-smi -q -d MEMORY \
|grep -A 2 "FB Memory"|grep Used \
|awk 'BEGIN{n=0;m=99999;a=0;}{if ($3<m) {m=$3;a=n;}; n++}END{printf("%d\n",a);}'
