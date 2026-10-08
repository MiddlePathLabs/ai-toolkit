'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { defaultJobConfig, defaultDatasetConfig, migrateJobConfig, pruneUntouchedLossBlocks } from './jobConfig';
import { resetProcessNameTags } from '@/helpers/nameTag';
import { jobTypeOptions } from './options';
import { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { useNestedState, setNestedValue } from '@/utils/hooks';
import { SelectInput } from '@/components/formInputs';
import useSettings from '@/hooks/useSettings';
import useGPUInfo from '@/hooks/useGPUInfo';
import useDatasetList from '@/hooks/useDatasetList';
import YAML from 'yaml';
import path from 'path';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { FaChevronLeft } from 'react-icons/fa';
import SimpleJob from './SimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import AdmissionFeedback from '@/components/AdmissionFeedback';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';
import {
  createAdmissionValidator,
  formatAdmissionError,
  getAdmissionCandidateKey,
  getCurrentAdmissionValidation,
  type AdmissionResult,
  type AdmissionValidation,
  type AdmissionValidator,
} from '@/utils/admission';
const isDev = process.env.NODE_ENV === 'development';

export default function TrainingForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');
  const cloneId = searchParams.get('cloneId');
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const { settings, isSettingsLoaded } = useSettings();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const { datasets, status: datasetFetchStatus } = useDatasetList();
  const [datasetOptions, setDatasetOptions] = useState<{ value: string; label: string }[]>([]);
  const [showAdvancedView, setShowAdvancedView] = useState(false);

  const [jobConfig, setJobConfig] = useNestedState<JobConfig>(objectCopy(migrateJobConfig(defaultJobConfig)));
  const [status, setStatus] = useState<'idle' | 'validating' | 'saving' | 'success' | 'error'>('idle');
  const [admissionValidation, setAdmissionValidation] = useState<AdmissionValidation | null>(null);
  const [syntaxResult, setSyntaxResult] = useState<AdmissionResult | null>(null);
  const [rawYaml, setRawYaml] = useState<string | undefined>();
  const [rawYamlValid, setRawYamlValid] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const candidateRawYaml = showAdvancedView ? rawYaml : undefined;
  // Form saves strip loss/noising blocks still at their untouched disabled
  // defaults (the trainer treats a missing block as off; migrateJobConfig
  // re-merges them on load). The pruned form is what admission validates and
  // what is saved, so the two never diverge. Raw YAML edits are kept verbatim.
  const candidateConfig = useMemo(
    () => (candidateRawYaml === undefined ? pruneUntouchedLossBlocks(jobConfig) : jobConfig),
    [jobConfig, candidateRawYaml],
  );
  const candidateKey = useMemo(
    () => getAdmissionCandidateKey(candidateConfig, candidateRawYaml),
    [candidateConfig, candidateRawYaml],
  );
  const candidateKeyRef = useRef(candidateKey);
  candidateKeyRef.current = candidateKey;
  const validatorRef = useRef<AdmissionValidator | null>(null);
  if (!validatorRef.current) validatorRef.current = createAdmissionValidator(setAdmissionValidation);
  const validator = validatorRef.current;
  validator.setCandidate(candidateKey);
  const validateCandidate = validator.validate;
  const currentValidation = getCurrentAdmissionValidation(admissionValidation, candidateKey);
  const admissionResult = rawYamlValid ? currentValidation?.result ?? null : syntaxResult;
  const admissionUnavailable = rawYamlValid ? currentValidation?.unavailable ?? null : null;
  const admissionStatus = rawYamlValid ? currentValidation?.status ?? 'pending' : 'invalid';
  const canSave = rawYamlValid && (admissionStatus === 'valid' || admissionStatus === 'runtime_pending');

  const handleImportConfig = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const text = reader.result as string;
      try {
        const parsed =
          file.name.endsWith('.json') || file.name.endsWith('.jsonc')
            ? JSON.parse(text.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, ''))
            : YAML.parse(text);
        const candidate = resetProcessNameTags(migrateJobConfig(parsed as JobConfig));
        setRawYaml(undefined);
        setRawYamlValid(true);
        setSyntaxResult(null);
        setJobConfig(candidate);
      } catch (error) {
        validator.invalidate();
        setRawYamlValid(false);
        setSyntaxResult({
          valid: false,
          diagnostics: [
            {
              rule_id: 'admission.raw_yaml_invalid',
              severity: 'error',
              fields: ['config'],
              reason: `The imported configuration could not be parsed: ${formatAdmissionError(error)}`,
              remedy: 'Fix the imported YAML/JSON syntax and import it again.',
              phase: 'static',
            },
          ],
          deferred: [],
          source: 'ui',
        });
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  useEffect(() => {
    if (!isSettingsLoaded) return;
    if (datasetFetchStatus !== 'success') return;

    const datasetOptions = datasets.map(name => ({ value: path.join(settings.DATASETS_FOLDER, name), label: name }));
    setDatasetOptions(datasetOptions);

    if (datasetOptions.length > 0) {
      const defaultDatasetPath = defaultDatasetConfig.folder_path;
      // Use functional updater so we check the *current* state, not a stale closure
      setJobConfig((prev: JobConfig) => {
        let updated = prev;
        for (let i = 0; i < prev.config.process[0].datasets.length; i++) {
          if (prev.config.process[0].datasets[i].folder_path === defaultDatasetPath) {
            updated = setNestedValue(updated, datasetOptions[0].value, `config.process[0].datasets[${i}].folder_path`);
          }
        }
        return updated;
      });
    }
  }, [datasets, settings, isSettingsLoaded, datasetFetchStatus]);

  useEffect(() => {
    const sourceId = cloneId || runId;
    if (!sourceId) return;
    let cancelled = false;
    apiClient
      .get(`/api/jobs?id=${sourceId}`)
      .then(res => {
        if (cancelled) return;
        const data = res.data;
        const loaded = resetProcessNameTags(migrateJobConfig(JSON.parse(data.job_config) as JobConfig));
        if (cloneId) loaded.config.name = `${loaded.config.name}_copy`;
        setGpuIDs(data.gpu_ids);
        setRawYaml(undefined);
        setRawYamlValid(true);
        setSyntaxResult(null);
        setJobConfig(loaded);
      })
      .catch(error => {
        if (cancelled) return;
        setAdmissionValidation({
          candidateKey: candidateKeyRef.current,
          status: 'unavailable',
          result: null,
          unavailable: formatAdmissionError(error),
        });
      });
    return () => { cancelled = true; };
  }, [cloneId, runId]);

  useEffect(() => {
    if (!rawYamlValid || status === 'validating' || status === 'saving') return;
    const handle = setTimeout(() => {
      void validateCandidate(candidateConfig, candidateRawYaml);
    }, 500);
    return () => clearTimeout(handle);
  }, [candidateKey, rawYamlValid, status]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  useEffect(() => {
    if (isSettingsLoaded) {
      setJobConfig(settings.TRAINING_FOLDER, 'config.process[0].training_folder');
    }
  }, [settings, isSettingsLoaded]);

  const saveJob = async () => {
    if (status === 'saving' || status === 'validating') return;
    if (!rawYamlValid) {
      setStatus('error');
      return;
    }
    const candidate = candidateConfig;
    const raw = candidateRawYaml;
    const key = candidateKey;
    setStatus('validating');
    const result = await validateCandidate(candidate, raw);
    if (!result || !result.valid || candidateKeyRef.current !== key) {
      setStatus('error');
      setTimeout(() => setStatus('idle'), 2000);
      return;
    }

    setStatus('saving');
    apiClient
      .post('/api/jobs', {
        id: runId,
        name: candidate.config.name,
        gpu_ids: gpuIDs,
        job_config: candidate,
        ...(raw === undefined ? {} : { raw_yaml: raw }),
      })
      .then(res => {
        setStatus('success');
        if (runId) router.push(`/jobs/${runId}`);
        else router.push(`/jobs/${res.data.id}`);
      })
      .catch(error => {
        if (error.response?.status === 409) alert('Training name already exists. Please choose a different name.');
        else alert(formatAdmissionError(error));
        setStatus('error');
      })
      .finally(() => setTimeout(() => setStatus('idle'), 2000));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    saveJob();
  };

  const admissionFeedback = (
    <AdmissionFeedback
      config={jobConfig}
      result={admissionResult}
      unavailable={admissionUnavailable}
      onRemediate={next => {
        setRawYaml(undefined);
        setJobConfig(next);
      }}
    />
  );

  return (
    <>
      <TopBar>
        <div className="flex-shrink-0">
          <Button className="text-gray-500 dark:text-gray-300 px-2 sm:px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="flex-shrink-0">
          <h1 className="text-base sm:text-lg truncate max-w-[120px] sm:max-w-none">
            {runId ? 'Edit Training Job' : 'New Training Job'}
          </h1>
        </div>
        <div className="flex-1"></div>
        {showAdvancedView && (
          <>
            <div className="hidden sm:block">
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
            <div className="hidden sm:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
            <div className="hidden md:block">
              <Button className="text-gray-200 bg-gray-800 px-3 py-1 rounded-md" onClick={handleImportConfig}>
                Import Config
              </Button>
            </div>
            <div className="hidden md:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
          </>
        )}
        {!showAdvancedView && (
          <>
            <div className="hidden sm:block">
              <SelectInput
                value={`${jobConfig?.config.process[0].type}`}
                onChange={value => {
                  // undo current job type changes
                  const currentOption = jobTypeOptions.find(
                    option => option.value === jobConfig?.config.process[0].type,
                  );
                  if (currentOption && currentOption.onDeactivate) {
                    setJobConfig(currentOption.onDeactivate(objectCopy(jobConfig)));
                  }
                  const option = jobTypeOptions.find(option => option.value === value);
                  if (option) {
                    if (option.onActivate) {
                      setJobConfig(option.onActivate(objectCopy(jobConfig)));
                    }
                    jobTypeOptions.forEach(opt => {
                      if (opt.value !== option.value && opt.onDeactivate) {
                        setJobConfig(opt.onDeactivate(objectCopy(jobConfig)));
                      }
                    });
                  }
                  setJobConfig(value, 'config.process[0].type');
                }}
                options={jobTypeOptions}
              />
            </div>
            <div className="hidden sm:block mx-4 bg-gray-200 dark:bg-gray-800 w-1 h-6"></div>
          </>
        )}

        <div className="pr-1 sm:pr-2 flex-shrink-0">
          <Button
            className="text-gray-200 bg-gray-800 px-2 sm:px-3 py-1 rounded-md text-xs sm:text-base"
            onClick={() => {
              setRawYaml(undefined);
              setShowAdvancedView(!showAdvancedView);
            }}
          >
            <span className="sm:hidden">{showAdvancedView ? 'Simple' : 'Advanced'}</span>
            <span className="hidden sm:inline">{showAdvancedView ? 'Show Simple' : 'Show Advanced'}</span>
          </Button>
        </div>
        <div className="flex-shrink-0">
          <Button
            className="text-white bg-green-600 hover:bg-green-700 px-2 sm:px-3 py-1 rounded-md text-xs sm:text-base"
            onClick={() => saveJob()}
            disabled={status === 'saving' || status === 'validating' || !canSave}
          >
            {status === 'saving' ? (
              'Saving...'
            ) : status === 'validating' ? (
              'Validating...'
            ) : (
              <>
                <span className="sm:hidden">{runId ? 'Update' : 'Create'}</span>
                <span className="hidden sm:inline">{runId ? 'Update Job' : 'Create Job'}</span>
              </>
            )}
          </Button>
        </div>
      </TopBar>

      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml,.json,.jsonc"
        style={{ display: 'none' }}
        onChange={handleFileSelected}
      />


      {showAdvancedView ? (
        <div className="absolute top-12 bottom-0 left-0 w-full overflow-y-auto">
          <div className="pb-3">{admissionFeedback}</div>
          <div className="h-[calc(100vh-3rem)] min-h-96">
          <AdvancedConfigEditor
            config={jobConfig}
            setConfig={setJobConfig}
            onRawChange={setRawYaml}
            onValidationChange={(valid, message, line) => {
              setRawYamlValid(valid);
              if (!valid) {
                validator.invalidate();
                setSyntaxResult({
                  valid: false,
                  diagnostics: [
                    {
                      rule_id: 'admission.raw_yaml_invalid',
                      severity: 'error',
                      fields: ['config'],
                      reason: `${message || 'Invalid YAML'} (line ${line || 1}).`,
                      remedy: 'Fix the highlighted YAML syntax before saving or starting this job.',
                      phase: 'static',
                    },
                  ],
                  deferred: [],
                  source: 'ui',
                });
              } else {
                setSyntaxResult(null);
              }
            }}
            transformOnParse={(parsed: any) => {
              try {
                parsed.config.process[0].sqlite_db_path = './aitk_db.db';
                parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
                parsed.config.process[0].performance_log_every = 10;
              } catch (e) {
                console.warn(e);
              }
              return migrateJobConfig(parsed);
            }}
          />
          </div>
        </div>
      ) : (
        <MainContent>
          {admissionFeedback}
          <ErrorBoundary
            fallback={
              <div className="flex items-center justify-center h-64 text-lg text-red-600 font-medium bg-red-100 dark:bg-red-900/20 dark:text-red-400 border border-red-300 dark:border-red-700 rounded-lg">
                Advanced job detected. Please switch to advanced view to continue.
              </div>
            }
          >
            <SimpleJob
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              status={status}
              handleSubmit={handleSubmit}
              runId={runId}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              datasetOptions={datasetOptions}
              isLoading={!isSettingsLoaded || !isGPUInfoLoaded || datasetFetchStatus !== 'success'}
            />
          </ErrorBoundary>

          <div className="pt-20"></div>
        </MainContent>
      )}
    </>
  );
}
