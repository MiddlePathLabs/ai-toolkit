import React, { useCallback, useLayoutEffect, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { Plus, RefreshCw, X } from 'lucide-react';
import { apiClient } from '@/utils/api';
import LoraBrowserModal, { LoraPick } from '@/components/generate/LoraBrowserModal';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  NumberInput,
  SelectInput,
  SliderInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import { CaptionJobConfig, CaptionLora } from '@/types';
import { handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
import {
  batchSizeOptions,
  captionFormatOptions,
  captionerTypes,
  concurrencyOptions,
  videoFramesOptions,
  defaultQtype,
  groupedCaptionerTypes,
  maxNewTokensOptions,
  maxResOptions,
  mtpDepthOptions,
  quantizationOptions,
  videoFpsOptions,
} from '@/helpers/captionOptions';

type Props = {
  jobConfig: CaptionJobConfig;
  setJobConfig: (value: any, key?: string) => void;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  showGPUSelect: boolean;
};

const CaptionSimpleJob: React.FC<Props> = ({ jobConfig, setJobConfig, gpuIDs, setGpuIDs, gpuList, showGPUSelect }) => {
  const selectedCaptionOption = captionerTypes.find(option => option.name === jobConfig.config.process[0].type);
  const additionalSections = selectedCaptionOption?.additionalSections || [];
  const captionPrompts = selectedCaptionOption?.captionPrompts || {};
  const promptPresetNames = Object.keys(captionPrompts);
  const minNewTokens = selectedCaptionOption?.minNewTokens ?? 0;
  const newTokensOptions = maxNewTokensOptions.filter(option => parseInt(option.value) >= minNewTokens);
  const [loraModalOpen, setLoraModalOpen] = useState(false);
  const loras: CaptionLora[] = jobConfig.config.process[0].caption.loras || [];
  const setLoras = (next: CaptionLora[]) => setJobConfig(next, 'config.process[0].caption.loras');
  const addLora = (pick: LoraPick) => {
    if (loras.some(l => l.path === pick.path)) return;
    setLoras([...loras, { path: pick.path, name: pick.name, strength: 1.0 }]);
  };
  const isRemote = !!selectedCaptionOption?.isRemote;
  const caption = jobConfig.config.process[0].caption;
  const [remoteModels, setRemoteModels] = useState<string[]>([]);
  const [modelsStatus, setModelsStatus] = useState<string | null>(null);
  const processType = jobConfig.config.process[0].type;
  const currentProcess = useRef(jobConfig.config.process[0]);
  const modelRequest = useRef<AbortController | null>(null);
  const modelEdited = useRef(false);
  const invalidateRemoteModels = useCallback(() => {
    modelRequest.current?.abort();
    modelRequest.current = null;
    setRemoteModels([]);
    setModelsStatus(null);
  }, []);
  useLayoutEffect(() => {
    currentProcess.current = jobConfig.config.process[0];
  });
  useLayoutEffect(() => {
    invalidateRemoteModels();
    // Modal returns null when closed, unmounting this form. Abort there (and
    // when switching to Advanced) so reopening cannot receive the old result.
    return () => {
      modelRequest.current?.abort();
      modelRequest.current = null;
    };
  }, [processType, caption.api_base_url, caption.api_key, invalidateRemoteModels]);
  const fetchRemoteModels = async () => {
    if (!isRemote) return;
    modelRequest.current?.abort();
    const controller = new AbortController();
    modelRequest.current = controller;
    modelEdited.current = false;
    const isCurrent = () => {
      const current = currentProcess.current;
      return modelRequest.current === controller && !controller.signal.aborted
        && current.type === processType
        && current.caption.api_base_url === caption.api_base_url
        && current.caption.api_key === caption.api_key;
    };
    setModelsStatus('Fetching models...');
    try {
      const res = await apiClient.post('/api/caption/models', {
        api_base_url: caption.api_base_url,
        api_key: caption.api_key,
      }, { signal: controller.signal });
      if (!isCurrent()) return;
      const models: string[] = res.data?.models || [];
      setRemoteModels(models);
      setModelsStatus(models.length ? `${models.length} model(s) found` : 'Endpoint returned no models');
      if (models.length && !modelEdited.current && !currentProcess.current.caption.model_name_or_path) {
        setJobConfig(models[0], 'config.process[0].caption.model_name_or_path');
      }
    } catch (e: unknown) {
      if (!isCurrent()) return;
      let reason = e instanceof Error ? e.message : 'request failed';
      if (isAxiosError(e)) {
        const data: unknown = e.response?.data;
        if (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string') {
          reason = data.error;
        }
      }
      setModelsStatus(`Could not list models: ${reason}`);
    } finally {
      if (modelRequest.current === controller) modelRequest.current = null;
    }
  };
  const modelOptions = isRemote
    ? remoteModels.map(m => ({ value: m, label: m }))
    : selectedCaptionOption?.name_or_path_options || [];

  return (
    <div className="text-sm text-gray-400">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="Captioner Type"
            value={jobConfig.config.process[0].type}
            onChange={value => {
              invalidateRemoteModels();
              handleCaptionerTypeChange(jobConfig.config.process[0].type, value, jobConfig, setJobConfig);
            }}
            options={groupedCaptionerTypes}
          />
        </div>
        {showGPUSelect && (
          <div>
            <SelectInput
              label="GPU ID"
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
      </div>
      {additionalSections.includes('caption.api_endpoint') && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <TextInput
            label="API Base URL"
            docKey="config.process[0].caption.api_base_url"
            value={caption.api_base_url || ''}
            onChange={value => {
              invalidateRemoteModels();
              setJobConfig(value.trim() === '' ? undefined : value, 'config.process[0].caption.api_base_url');
            }}
            placeholder="http://localhost:8080/v1"
            required
          />
          <TextInput
            label="API Key (optional)"
            type="password"
            value={caption.api_key || ''}
            onChange={value => {
              invalidateRemoteModels();
              setJobConfig(value.trim() === '' ? undefined : value, 'config.process[0].caption.api_key');
            }}
            placeholder="Blank: job uses AITK_CAPTION_API_KEY env var"
          />
        </div>
      )}
      <div className="mt-4">
        <CreatableSelectInput
          label={isRemote ? 'Model Name' : 'Name or Path'}
          value={caption.model_name_or_path}
          docKey={isRemote ? 'config.process[0].caption.api_model' : 'config.process[0].caption.model_name_or_path'}
          onChange={(value: string | null) => {
            modelEdited.current = true;
            if (value?.trim() === '') {
              value = null;
            }
            setJobConfig(value, 'config.process[0].caption.model_name_or_path');
          }}
          placeholder={isRemote ? 'Model name the server expects' : ''}
          options={modelOptions}
          required
        />
        {isRemote && (
          <div className="flex items-center gap-2 mt-1">
            <button
              type="button"
              onClick={fetchRemoteModels}
              disabled={!caption.api_base_url}
              className="px-2 py-1 rounded-md bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs flex items-center gap-1 disabled:opacity-40"
            >
              <RefreshCw className="w-3.5 h-3.5" /> List models from endpoint
            </button>
            {modelsStatus && <span className="text-[11px] text-gray-500">{modelsStatus}</span>}
          </div>
        )}
      </div>
      {additionalSections.includes('caption.model_name_or_path2') && (
        <div className="mt-4">
          <CreatableSelectInput
            label="Name or Path 2"
            value={jobConfig.config.process[0].caption.model_name_or_path2 || ''}
            onChange={(value: string | null) => {
              if (value?.trim() === '') {
                value = null;
              }
              setJobConfig(value, 'config.process[0].caption.model_name_or_path2');
            }}
            placeholder=""
            options={selectedCaptionOption?.name_or_path2_options || []}
          />
        </div>
      )}
      {additionalSections.includes('caption.caption_format') && (
        <div className="mt-4">
          <SelectInput
            label="Caption Format"
            value={jobConfig.config.process[0].caption.caption_format || 'ace_step'}
            onChange={value => setJobConfig(value, 'config.process[0].caption.caption_format')}
            options={captionFormatOptions}
          />
        </div>
      )}
      {additionalSections.includes('caption.fixed_caption') && (
        <div className="mt-4">
          <TextInput
            label="Fixed Caption"
            value={jobConfig.config.process[0].caption.fixed_caption || ''}
            onChange={value => {
              if (value?.trim() === '') {
                //@ts-ignore
                value = undefined;
              }
              setJobConfig(value, 'config.process[0].caption.fixed_caption');
            }}
            placeholder="Enter fixed caption (if you want the same caption for all audio files)"
          />
        </div>
      )}
      {selectedCaptionOption?.supportsLoras && (
        <div className="mt-4">
          <div className="text-xs text-gray-300 mb-1">LoRAs</div>
          <div className="space-y-2">
            {loras.map((l, i) => (
              <div key={l.path} className="bg-gray-950/60 border border-gray-800 rounded-md px-2 py-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs truncate flex-1 text-gray-200" title={l.path}>
                    {l.name}
                  </span>
                  <input
                    type="number"
                    step={0.05}
                    min={-2}
                    max={3}
                    value={l.strength}
                    onChange={e => {
                      const v = parseFloat(e.target.value);
                      const next = [...loras];
                      next[i] = { ...l, strength: isNaN(v) ? 0 : v };
                      setLoras(next);
                    }}
                    className="w-16 text-xs px-1.5 py-0.5 bg-gray-950 border border-gray-700 rounded text-gray-100 text-right"
                  />
                  <button
                    type="button"
                    onClick={() => setLoras(loras.filter(x => x.path !== l.path))}
                    className="text-gray-500 hover:text-red-400"
                    title="Remove"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() => setLoraModalOpen(true)}
              className="w-full px-2 py-1 rounded-md bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs flex items-center justify-center gap-1"
            >
              <Plus className="w-3.5 h-3.5" /> Add LoRA
            </button>
          </div>
          <div className="text-[11px] text-gray-500 mt-1">
            Applied as sidechains at caption time; the model weights are never merged.
          </div>
          <LoraBrowserModal
            isOpen={loraModalOpen}
            onClose={() => setLoraModalOpen(false)}
            onPick={addLora}
            cloudLoras={selectedCaptionOption?.cloudLoras}
          />
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          {!isRemote && (
            <SelectInput
              label="Quantize"
              value={jobConfig.config.process[0].caption.quantize ? jobConfig.config.process[0].caption.qtype : ''}
              onChange={value => {
                if (value === '') {
                  setJobConfig(false, 'config.process[0].caption.quantize');
                  value = defaultQtype;
                } else {
                  setJobConfig(true, 'config.process[0].caption.quantize');
                }
                setJobConfig(value, 'config.process[0].caption.qtype');
              }}
              options={quantizationOptions}
            />
          )}
          {isRemote && (
            <SelectInput
              label="Parallel Requests"
              value={`${caption.concurrency ?? 2}`}
              onChange={value => {
                const intVal = parseInt(value);
                if (!isNaN(intVal)) setJobConfig(intVal, 'config.process[0].caption.concurrency');
              }}
              options={concurrencyOptions}
            />
          )}
          <div className="mt-4">
            <CreatableSelectInput
              label="Caption Extension"
              value={jobConfig.config.process[0].caption.caption_extension || 'txt'}
              onChange={value => {
                setJobConfig(value, 'config.process[0].caption.caption_extension');
              }}
              options={[
                { value: 'txt', label: 'txt' },
                { value: 'json', label: 'json' },
                { value: 'caption', label: 'caption' },
              ]}
            />
          </div>
          {additionalSections.includes('caption.max_res') && (
            <div className="mt-4">
              <SelectInput
                label="Max Resolution"
                value={`${jobConfig.config.process[0].caption.max_res || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_res');
                  }
                }}
                options={maxResOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.max_new_tokens') && (
            <div className="mt-4">
              <SelectInput
                label="Max New Tokens"
                value={`${jobConfig.config.process[0].caption.max_new_tokens || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_new_tokens');
                  }
                }}
                options={newTokensOptions}
              />
            </div>
          )}
          {isRemote && (
            <>
              <div className="mt-4">
                <SelectInput
                  label="Frames per Video"
                  value={`${caption.video_frames ?? 8}`}
                  onChange={value => {
                    const intVal = parseInt(value);
                    if (!isNaN(intVal)) setJobConfig(intVal, 'config.process[0].caption.video_frames');
                  }}
                  options={videoFramesOptions}
                />
              </div>
              <div className="mt-4">
                <NumberInput
                  label="Temperature"
                  value={caption.temperature ?? 0.2}
                  onChange={value => setJobConfig(value, 'config.process[0].caption.temperature')}
                  min={0}
                  max={2}
                />
              </div>
            </>
          )}
          {additionalSections.includes('caption.video_fps') && (
            <div className="mt-4">
              <SelectInput
                label="Video FPS"
                value={`${jobConfig.config.process[0].caption.video_fps ?? 2}`}
                onChange={value => {
                  const num = parseFloat(value);
                  if (!isNaN(num)) {
                    setJobConfig(num, 'config.process[0].caption.video_fps');
                  }
                }}
                options={videoFpsOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.batch_size') && (
            <div className="mt-4">
              <SelectInput
                label="Batch Size"
                value={`${jobConfig.config.process[0].caption.batch_size || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.batch_size');
                  }
                }}
                options={batchSizeOptions}
              />
            </div>
          )}
        </div>
        <div>
          <FormGroup label="Options">
            {!isRemote && (
              <Checkbox
                label="Low VRAM"
                checked={jobConfig.config.process[0].caption.low_vram}
                onChange={value => setJobConfig(value, 'config.process[0].caption.low_vram')}
              />
            )}
            <Checkbox
              label="Recaption"
              checked={jobConfig.config.process[0].caption.recaption}
              onChange={value => setJobConfig(value, 'config.process[0].caption.recaption')}
            />
            {!selectedCaptionOption?.noCompile && (
              <Checkbox
                label="Compile Models"
                checked={jobConfig.config.process[0].caption.compile || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.compile')}
              />
            )}
            {additionalSections.includes('caption.extract_vocals_before_transcribe') && (
              <Checkbox
                label="Extract Vocals Before Transcribing"
                checked={jobConfig.config.process[0].caption.extract_vocals_before_transcribe || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.extract_vocals_before_transcribe')}
              />
            )}
            {additionalSections.includes('caption.keep_timestamps') && (
              <Checkbox
                label="Keep Lyric Timestamps"
                checked={jobConfig.config.process[0].caption.keep_timestamps || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.keep_timestamps')}
              />
            )}
            {additionalSections.includes('caption.thinking') && (
              <Checkbox
                label="Thinking"
                checked={jobConfig.config.process[0].caption.thinking || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.thinking')}
              />
            )}
            {additionalSections.includes('caption.mtp') && (
              <>
                <Checkbox
                  label="MTP Speculative Decoding"
                  checked={jobConfig.config.process[0].caption.mtp ?? true}
                  onChange={value => setJobConfig(value, 'config.process[0].caption.mtp')}
                />
                {(jobConfig.config.process[0].caption.mtp ?? true) && (
                  <div className="pt-2">
                    <SelectInput
                      label="MTP Draft Depth"
                      value={`${jobConfig.config.process[0].caption.mtp_depth ?? 3}`}
                      onChange={value => {
                        const intVal = parseInt(value);
                        if (!isNaN(intVal)) {
                          setJobConfig(intVal, 'config.process[0].caption.mtp_depth');
                        }
                      }}
                      options={mtpDepthOptions}
                    />
                  </div>
                )}
              </>
            )}
            {additionalSections.includes('caption.cuda_graph') && (
              <Checkbox
                label="CUDA Graph Decode"
                checked={jobConfig.config.process[0].caption.cuda_graph ?? true}
                onChange={value => setJobConfig(value, 'config.process[0].caption.cuda_graph')}
              />
            )}
            {additionalSections.includes('caption.vision_bf16') && (
              <Checkbox
                label="Vision Tower bf16"
                checked={jobConfig.config.process[0].caption.vision_bf16 ?? true}
                onChange={value => setJobConfig(value, 'config.process[0].caption.vision_bf16')}
              />
            )}
            {additionalSections.includes('caption.layer_offloading') && (
              <>
                <Checkbox
                  label="Layer Offloading"
                  checked={jobConfig.config.process[0].caption.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].caption.layer_offloading')}
                />
                {jobConfig.config.process[0].caption.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="Offload %"
                      value={Math.round((jobConfig.config.process[0].caption.layer_offloading_percent ?? 1) * 100)}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].caption.layer_offloading_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
              </>
            )}
          </FormGroup>
        </div>
      </div>
      {additionalSections.includes('caption.caption_prompt') && (
        <div className="mt-4">
          {promptPresetNames.length > 1 && (
            <div className="mb-4">
              <SelectInput
                label="Prompt Preset"
                value={
                  promptPresetNames.find(
                    name => captionPrompts[name] === jobConfig.config.process[0].caption.caption_prompt,
                  ) || ''
                }
                onChange={value => {
                  if (captionPrompts[value] !== undefined) {
                    setJobConfig(captionPrompts[value], 'config.process[0].caption.caption_prompt');
                  }
                }}
                options={[
                  { value: '', label: '- Custom -' },
                  ...promptPresetNames.map(name => ({ value: name, label: name })),
                ]}
              />
            </div>
          )}
          {additionalSections.includes('caption.system_prompt') && (
            <div className="mb-4">
              <TextInput
                label="System Prompt"
                value={jobConfig.config.process[0].caption.system_prompt || ''}
                onChange={value => {
                  setJobConfig(value?.trim() === '' ? undefined : value, 'config.process[0].caption.system_prompt');
                }}
                placeholder="Optional system message"
              />
            </div>
          )}
          <TextAreaInput
            label="Caption Prompt"
            docKey="config.process[0].caption.caption_prompt"
            value={jobConfig.config.process[0].caption.caption_prompt || ''}
            onChange={value => {
              setJobConfig(value, 'config.process[0].caption.caption_prompt');
            }}
            placeholder="Enter caption prompt"
          />
        </div>
      )}
    </div>
  );
};

export default CaptionSimpleJob;
