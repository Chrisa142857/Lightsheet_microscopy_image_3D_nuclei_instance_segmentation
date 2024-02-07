### Executible to test a whole brain

#### Config

Check path config in the main function `test.cpp`, change path config to 

Generate `.pt` modules using `run_whole_brain/export_cpp_module.py`, so that `test.cpp` can utilize those modules.

#### To build

Check `CMakeLists.txt` and install required packages. Then
```
sh build.sh
```

#### Usage under default config
```
build/test ${path_dir1} ${path_dir2} ${device} ${path_root} > log.log
```