from typing import Dict
import time
import json
from sync import Sync
from sync.common import Audio, GenerationOptions, Video, GenerationSegment, SegmentAudioInput
from sync.core.api_error import ApiError


class LipSyncProcessor():
    """
    Handles lip-syncing using Sync.so API.
    """
    def __init__(self, lipsync_api_key: str):
        self.client = Sync(api_key=lipsync_api_key,)       

    def poll_for_status(self, jobs, timeout=3600, interval=10):
        """
        Poll the API to check the status of submitted lip sync jobs.
        
        This method continuously checks the status of all pending jobs until they
        complete, fail, or the timeout is reached.
        
        Args:
            jobs (list): List of job IDs or tuples (index, job_id) to monitor
            timeout (int, optional): Maximum time in seconds to wait for all jobs.
                                    Defaults to 3600 seconds (60 minutes).
            interval (int, optional): Time in seconds between status checks.
                                     Defaults to 10 seconds.
                                     
        Returns:
            list: Results data for all completed jobs, including their status and
                 output information. Each result includes an 'idx' field matching 
                 the job's original index.
                 
        Raises:
            Exception: If the polling process times out before all jobs complete.
        """
        start_time = time.time()
        results = []
        pending_jobs = set(jobs)
        
        while pending_jobs and (time.time() - start_time < timeout):
            # Check each pending job
            for (i,job_id) in list(pending_jobs):
                try:
                    response = self.client.generations.get(
                        id=job_id,
                    )
                    
                    data = json.loads(response.json())
                    data['idx'] = i
                    status = data.get('status')
                    
                    # If job is complete, remove from pending list
                    if status == 'COMPLETED':
                        pending_jobs.remove((i,job_id))
                        # normalize key name so callers can rely on output_url
                        data['output_url'] = data.get('output_url') or data.get('outputUrl')
                        results.append(data)
                        print(f"Job {job_id} completed successfully.")
                    elif status == "FAILED":
                        print(f"Lipsync process failed for {job_id} with status: {status} and error: {data.get('error','')}")
                        pending_jobs.remove((i,job_id))
                        data['output_url'] = f'Job Status {status}'
                        results.append(data) 
                except ApiError as e:
                    print(f"Error checking status for job {job_id}: {e.status_code} {e.body}")
            
            # If jobs still pending, wait before next check
            if pending_jobs:
                print(f"Waiting for {len(pending_jobs)} jobs to complete. Next check in {interval} seconds.")
                time.sleep(interval)
        
        # Check for timed out jobs
        if pending_jobs:
            print(f"Polling process timed out waiting for jobs: {pending_jobs}")
            for (i,job_id) in list(pending_jobs):
                data = {'idx':i, 'output_url':'POLLING_TIME_OUT'}
                results.append(data)              
        return results

    def process_lip_sync(self, entry:Dict):
        """
        Perform lip-syncing on a video using a specified audio source.
        
        Submits a job to the Sync.so API with video and audio inputs, along with
        configuration options for the lip-syncing process.
        
        Args:
            entry (Dict): Dictionary containing the necessary input data with keys:
                - 'video': URL of the video file to process
                - 'audio': URL of the audio file to use for lip-syncing
                - 'segment_start': Start time (in seconds) of the video segment
                - 'segment_end': End time (in seconds) of the video segment
                - 'temperature' (optional): float 0-1, expressiveness of lip/facial
                  movement. Defaults to 0.5 (sync.so's own default) if not provided.
                - 'model_mode' (optional): 'lips' / 'face' / 'head' — only used when
                  lipsync_model is 'react-1'. Controls how much of the performance
                  react-1 is allowed to regenerate.
                - 'emotion' (optional): 'happy' / 'sad' / 'angry' / 'disgusted' /
                  'surprised' / 'neutral' — only used when lipsync_model is 'react-1'.
                
        Returns:
            dict: JSON response from the API containing the job ID and status
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        
        # Pull temperature from the entry if present (e.g. from the web form),
        # otherwise fall back to sync.so's own default of 0.5.
        temperature = entry.get('temperature', 0.5)
        if temperature is not None:
            temperature = float(temperature)

        options_kwargs = {
            'sync_mode': entry['sync_mode'],
            'temperature': temperature,
        }

        # react-1-only parameters — only sent when that model is selected.
        if entry.get('lipsync_model') == 'react-1':
            if entry.get('model_mode'):
                options_kwargs['model_mode'] = entry['model_mode']
            if entry.get('emotion'):
                options_kwargs['prompt'] = entry['emotion']

        try:
            options = GenerationOptions(**options_kwargs)
        except TypeError as e:
            # Installed syncsdk version may predate model_mode/prompt support.
            # Fall back to the fields we know are always supported, so react-1
            # requests still go through (without the extra controls) instead
            # of crashing outright. Worth running `pip install --upgrade syncsdk`
            # if you see this warning and want the new controls to actually apply.
            print(f"GenerationOptions rejected react-1 params ({e}); "
                  f"falling back to sync_mode/temperature only. "
                  f"Try: pip install --upgrade syncsdk")
            options = GenerationOptions(
                sync_mode=entry['sync_mode'],
                temperature=temperature,
            )

        try:
            if entry.get('extra_slots'):
                # Multiple name windows in ONE generation: one job, one poll.
                # Slot 1 is the main window; extra_slots holds slots 2..N, each
                # {'audio': url, 'start': s, 'end': e}. Uses the top-level
                # `segments` array (segments_secs is deprecated and can't carry
                # multiple audio inputs).
                audio_inputs = [Audio(url=entry['audio'], ref_id="audio_1")]
                segments = [GenerationSegment(
                    start_time=float(entry['segment_start']),
                    end_time=float(entry['segment_end']),
                    audio_input=SegmentAudioInput(ref_id="audio_1"),
                )]
                for n, slot in enumerate(entry['extra_slots'], start=2):
                    ref = f"audio_{n}"
                    audio_inputs.append(Audio(url=slot['audio'], ref_id=ref))
                    segments.append(GenerationSegment(
                        start_time=float(slot['start']),
                        end_time=float(slot['end']),
                        audio_input=SegmentAudioInput(ref_id=ref),
                    ))
                response = self.client.generations.create(
                    input=[Video(url=entry['video'])] + audio_inputs,
                    segments=segments,
                    model=entry['lipsync_model'],
                    options=options,
                )
            else:
                response = self.client.generations.create(
                    input=[
                        Video(
                            url=entry['video'],
                            segments_secs=[[entry['segment_start'], entry['segment_end']]],
                        ),
                        Audio(url=entry['audio']),
                    ],
                    model=entry['lipsync_model'],
                    options=options,
                )

            return json.loads(response.json())
        except ApiError as e:
            print(e.status_code)
            print(e.body)
            raise
        