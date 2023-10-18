#!/bin/bash

# Clone and install ACRONYM code
git clone https://github.com/NVlabs/acronym.git
cd acronym
pip install -r requirements.txt
pip install -e .
cd ..

# Clone and build Manifold
git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git
cd Manifold
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
