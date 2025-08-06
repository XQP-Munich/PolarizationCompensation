#!/bin/bash
#trap 'ssh -S ssh_socket_alice -O exit morty' EXIT
#ssh -f -N -M -S ssh_socket_alice -R 31415:127.0.0.1:31415 morty
ssh -N -M -S ssh_socket_alice -L 31415:127.0.0.1:31415 morty
