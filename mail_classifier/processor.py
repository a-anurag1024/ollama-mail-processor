from dataclasses import dataclass 
from datetime import datetime 
from pathlib import Path
import os 
import glob 
import copy
import time
import threading
import json
from tqdm import tqdm


@dataclass
class ProcessorConfig:
    
    # run details
    run_name: str = "mail_processor_1"
    start_date: datetime = datetime.now()
    force_run: bool = False # if True, the pre-exising run will be re-initialized
    
    # input data folder
    input_data_folder: str = str(Path("./mount/metadata"))
    
    # save data folder
    save_folder: str = str(Path("./mount/processed_mail_metadata"))

    # run limits for each processing
    max_retries: int = 3
    max_wait_time: int = 5  # in seconds
        
    # log_folder
    log_folder: str = str(Path("./mount/mail_processing_logs"))


class TimeoutException(Exception):
    pass


class RunWithTimeout():
    def __init__(self, function, args):
        self.function = function
        self.args = args
        self.answer = None

    def worker(self):
        self.answer = self.function(*self.args)

    def run(self, timeout):
        thread = threading.Thread(target=self.worker)
        thread.start()
        thread.join(timeout)
        return self.answer


class Processor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        
        self.objects = None
        self.completed = None
        self._get_run_details()
        
        # initialize the alarm signal
        # signal.signal(signal.SIGALRM, timeout_handler)

        
    def _save_run_details(self):
        run_details = copy.deepcopy(self.config.__dict__)
        run_details["start_date"] = run_details["start_date"].strftime("%Y-%m-%d %H:%M:%S")
        
        for k, v in run_details.items():
            
            if type(v) not in [str, int, float, list, bool]:
                run_details[k] = v.__dict__
        
        with open(os.path.join(self.config.log_folder, "run_details.json"), "w") as f:
            json.dump(run_details, f, indent=3)
        
    def _initialize_run(self):
        if os.path.exists(os.path.join(self.config.log_folder, "run_details.json")):
            print(f" ||>> Run already initialized. The run details are kept at: {os.path.join(self.config.log_folder, 'run_details.json')}")
            if self.config.force_run:
                print(" ||>> [!!] Force run is enabled. The run will be re-initialized.")
                os.system(f"rm -rf {self.config.log_folder}")
            else:
                return False 
        
        os.makedirs(self.config.log_folder, exist_ok=True)
        self._save_run_details()
            
        os.makedirs(self.config.save_folder, exist_ok=True)
        return True
    
    
    def _get_run_details(self):
        init = self._initialize_run()
        
        if init:
            self.objects = self.get_run_input_objects()
            self.create_run_plan()
        else:
            self.objects = self.resume_run_plan()
    
    
    def get_run_input_objects(self):
        """
        function to be implemented in the child class to returns list of objects to be processed.
        - The input object dictionary should contain all the necessary input variables to the process function
        - Also the ordering should be as per the processing order
        
        Returns:
        - objects: List[Dict]: List of objects to be processed
        """
        return NotImplementedError("The function get_run_input_objects should be implemented in the child class")
    
    
    def create_run_plan(self):
        """
        from the ordered input objects list, this function creates the run plan
        """
        
        self.objects = [{"queue_id": i, **obj} for i, obj in enumerate(self.objects)]
        
        with open(os.path.join(self.config.log_folder, "run_plan.json"), "w") as f:
            json.dump(self.objects, f, indent=1)
            
        print(f"||>> Run plan created at: {os.path.join(self.config.log_folder, 'run_plan.json')}")
        print(f"||>> Total objects to be processed: {len(self.objects)}")
    
    
    def resume_run_plan(self):
        """
        function to resume the pre-existing run plan
        
        Returns:
        - objects: List[Dict]: List of objects to be processed
        """
        
        with open(os.path.join(self.config.log_folder, "run_plan.json"), "r") as f:
            objects = json.load(f)
        
        if not os.path.exists(os.path.join(self.config.log_folder, "completed.log")):
            completed = []
        else:
            with open(os.path.join(self.config.log_folder, "completed.log"), "r") as f:
                completed = [int(l.split('|')[0]) for l in f.readlines()]
    
        
        objects = [obj for obj in objects if obj["queue_id"] not in completed]
        print(f" ||>> Already processed {len(completed)} objects. Resuming the run from {(len(completed))+1}")
        print(f" ||>> Total objects to be processed: {len(objects)}")
        
        return objects
    
    
    def process(self, input_data):
        """
        Define this function in the child class such that it processes one of the object (input data) in the self.objects list
        
        Returns:
        - output: Dict: The output of the processing function
        """
        return NotImplementedError("The function get_run_input_objects should be implemented in the child class")
    
    
    
    def scheduler(self, input_data):
        """
        The scheduler function to run the process function with retries and wait times
        
        Args:
        - input_data: Dict: The input data to be processed
        
        Returns:
        - output: Any: The output of the process function
        - time_taken: float: The time taken to process the input data
        """
        
        retries = 0
        st = time.time()
        while retries < self.config.max_retries:
            try:
                process_thread = RunWithTimeout(self.process, (input_data,))
                process_ret = process_thread.run(self.config.max_wait_time)
                if process_ret is None:
                    raise TimeoutException(" [!] Process Timed out")
                en = time.time()
                return process_ret, en-st
            except TimeoutException as te:
                print(f" ||>> Error processing object {input_data['queue_id']} with error {te}. Retrying {retries}/{self.config.max_retries}")
                retries += 1
        raise Exception(f" [!!] Error processing object {input_data['queue_id']}. Max retries reached)")
    
    
    def run(self):
        """
        The main function to run the processor
        """
        
        for obj in tqdm(self.objects):
            try:
                i = obj["queue_id"]
                outputs, time_taken = self.scheduler(obj)
            except Exception as e:
                print(f" ||>> Error processing object {i+1}/{len(self.objects)}")
                print(f" ||>> Error: {e}")
                with open(os.path.join(self.config.log_folder, "error_log.json"), "a") as f:
                    json.dump({"queue_id": i, "error": str(e)}, f)
                    f.write("\n")
                continue
            with open(os.path.join(self.config.save_folder, f"{i}.json"), "w") as f:
                json.dump(outputs, f, indent=2)
            with open(os.path.join(self.config.log_folder, "completed.log"), "a") as f:
                f.write(str(i)+f"|{time_taken}\n")
                
        print(" ||>> Processing complete")
        print(" ||>> Run logs are kept at: ", self.config.log_folder)