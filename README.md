# sam2_annotator  
A webapp interface to SAM2 for labelling batches of image data. SAM2 enables multi-label object masking, and tracking across subsequent video frames. Supports exporting masks, and YOLO style bounding box annotations.    
![Tool GUI Image](/img/gui.png "Tool GUI")  


### Startup  
1. Run the container script using the full path of your data folder as the first argument. If you haven't built the container already, this script will do it automatically so it may take a while to run for the first time. It will mount your input data folder to '/workspace/data' within the container. Alternatively, you can change the *DEFAULT_LOCAL_DATA_DIR* variable inside the script, and use no command line arugments.
```
    ./run_sam_container.sh /path/to/data   
```
2. Modify the config/annotator_config.json file. This file allows you to specify model configuration, object labels, and a default 'data_folder' to load. (Coming Soon: I will implement a regex for matching image files of a specified pattern to load into the model). Different data folders may also be loaded while the webapp is running using the interface. *NOTE:* check how your data is mounted by calling ```ls /workspace/data```.

3. Run the Annotator Webapp script:  
```
    python3 scripts/annotator_webapp.py  
``` 

4. Open the address listed in the terminal in your web browser and you should see the annotator tool displayed.   

### Usage   
* Using the controls at the top of the page, you can specify a new input folder of images to load while the app is running.   
* Previous, Next, and Jump (to N) buttons allow you to step through the loaded images.   
* Select the object you're currently labelling with the 'Select Label' dropdown.   
* Left-Clicking on the image will add a positive prompt for the selected object. The 'negative prompt' checkbox will make it a negative prompt instead.   
* You can propagate/track masks across frames using the 'Propagate All From Here' and 'Propagate N From Here' buttons. Neither of these buttons will change any of the masks in previous frames. **If you need to make corrections on any frame, you should add the appropriate prompts, then re-propagate from that frame.**  
* Once you have all the masks you want, use the export buttons to save everything.  

### Exports    
* **YOLO Annotations** - Exports YOLO style bounding box annotation files. By default this will export them with the object NAME as the label rather than the internal numeric object ID (which is expected by YOLO). TODO: Make this optional.  
* **Binary Masks** - This will export black and white masks for each loaded images. By default, the object ID is encoded into the masks such that the mask color is 255-{obj_id}.   
* **Colored Masks** - This will export all images with their colored masks overlaid (same as how each frame is displayed within the interface).   
