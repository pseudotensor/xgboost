rm -rf build ; mkdir build ; cd build  ; cmake -DCMAKE_BUILD_TYPE=Debug  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  -DPLUGIN_UPDATER_GPU=ON -DCUB_DIRECTORY="~/cub/" .. ; make -j &> make0.log ; make clean &> make1.log ; make -j &> make.log
cd ../python-package
python setup.py install --user
python3 setup.py install --user
