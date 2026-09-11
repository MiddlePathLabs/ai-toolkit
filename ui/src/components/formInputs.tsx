'use client';

import React, { forwardRef } from 'react';
import classNames from 'classnames';
import dynamic from 'next/dynamic';
import { CircleHelp, ChevronDown } from 'lucide-react';
import { getDoc } from '@/docs';
import { openDoc } from '@/components/DocModal';
import { ConfigDoc, GroupedSelectOption, SelectOption } from '@/types';

const Select = dynamic(() => import('react-select'), { ssr: false });

const labelClasses = 'block text-xs mb-1 mt-2 text-gray-300';
const inputClasses =
  'w-full text-sm px-3 py-1 bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm text-gray-100 placeholder:text-gray-500 focus:ring-2 focus:ring-gray-600 focus:border-transparent';


function HoverTip({ text, children }: { text: string; children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const timer = React.useRef(0);
  const show = () => {
    timer.current = window.setTimeout(() => setOpen(true), 350);
  };
  const hide = () => {
    clearTimeout(timer.current);
    timer.current = 0;
    setOpen(false);
  };
  React.useEffect(() => () => clearTimeout(timer.current), []);
  return (
    <span className="relative inline" onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      {open && (
        <span
          role="tooltip"
          className="absolute left-0 top-full z-50 mt-1 w-64 rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-left text-xs font-normal normal-case tracking-normal text-gray-200 shadow-sm"
        >
          {text}
        </span>
      )}
    </span>
  );
}

function FieldLabel({
  label,
  doc,
  className,
}: {
  label?: React.ReactNode;
  doc?: ConfigDoc | null;
  className?: string;
}) {
  if (!label) return null;
  const summary = doc?.summary;
  const labelNode = summary ? (
    <HoverTip text={summary}>
      <span className="cursor-help border-b border-dotted border-gray-600">{label}</span>
    </HoverTip>
  ) : (
    label
  );
  return (
    <label className={classNames(labelClasses, className)}>
      {labelNode}
      {doc && (
        <button
          type="button"
          className="ml-1 inline-block align-middle text-gray-500 hover:text-gray-300"
          onClick={() => openDoc(doc)}
          aria-label="More about this setting"
        >
          <CircleHelp className="inline-block h-4 w-4" />
        </button>
      )}
    </label>
  );
}

function resolveDoc(doc?: ConfigDoc | null, docKey?: string | null): ConfigDoc | null {
  if (doc) return doc;
  if (docKey) return getDoc(docKey);
  return null;
}

export function AdvancedFold({ hint, children }: { hint?: string; children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="mt-3 border-t border-gray-800 pt-2">
      <button
        type="button"
        className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300"
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
      >
        <ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />
        Advanced
        {!open && hint ? <span className="text-gray-600">· {hint}</span> : null}
      </button>
      {open && <div className="pt-2">{children}</div>}
    </div>
  );
}

export interface InputProps {
  label?: string;
  docKey?: string | null;
  doc?: ConfigDoc | null;
  className?: string;
  placeholder?: string;
  required?: boolean;
}

export interface TextInputProps extends InputProps {
  value: string;
  onChange: (value: string) => void;
  type?: 'text' | 'password';
  disabled?: boolean;
  suffix?: React.ReactNode;
}

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>((props: TextInputProps, ref) => {
  const {
    label,
    value,
    onChange,
    placeholder,
    required,
    disabled,
    type = 'text',
    className,
    docKey = null,
    suffix,
  } = props;
  const doc = resolveDoc(props.doc, docKey);
  return (
    <div className={classNames(className)}>
      <FieldLabel label={label} doc={doc} />
      {suffix ? (
        <div
          className={classNames(
            'flex items-stretch w-full bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm focus-within:ring-2 focus-within:ring-gray-600',
            disabled ? 'opacity-30 cursor-not-allowed' : '',
          )}
        >
          <input
            ref={ref}
            type={type}
            value={value}
            onChange={e => {
              if (!disabled) onChange(e.target.value);
            }}
            className="flex-1 min-w-0 bg-transparent text-sm px-3 py-1 text-gray-100 placeholder:text-gray-500 focus:outline-none"
            placeholder={placeholder}
            required={required}
            disabled={disabled}
          />
          <span className="flex items-center px-2 text-sm text-gray-400 border-l border-gray-700 bg-gray-900/50 select-none">
            {suffix}
          </span>
        </div>
      ) : (
        <input
          ref={ref}
          type={type}
          value={value}
          onChange={e => {
            if (!disabled) onChange(e.target.value);
          }}
          className={`${inputClasses} ${disabled ? 'opacity-30 cursor-not-allowed' : ''}`}
          placeholder={placeholder}
          required={required}
          disabled={disabled}
        />
      )}
    </div>
  );
});

// 👇 Helpful for debugging
TextInput.displayName = 'TextInput';

export interface TextAreaInputProps extends InputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  rows?: number;
}

export const TextAreaInput = forwardRef<HTMLTextAreaElement, TextAreaInputProps>((props: TextAreaInputProps, ref) => {
  const { label, value, onChange, placeholder, required, disabled, rows = 4, className, docKey = null } = props;
  const doc = resolveDoc(props.doc, docKey);
  return (
    <div className={classNames(className)}>
      <FieldLabel label={label} doc={doc} />
      <textarea
        ref={ref}
        value={value}
        onChange={e => {
          if (!disabled) onChange(e.target.value);
        }}
        className={`${inputClasses} ${disabled ? 'opacity-30 cursor-not-allowed' : ''}`}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
        rows={rows}
      />
    </div>
  );
});

TextAreaInput.displayName = 'TextAreaInput';

export interface NumberInputProps extends InputProps {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  // when true, clearing the input calls onChange(null) instead of being ignored
  allowEmpty?: boolean;
}

export const NumberInput = (props: NumberInputProps) => {
  const { label, value, onChange, placeholder, required, min, max, allowEmpty, docKey = null } = props;
  const doc = resolveDoc(props.doc, docKey);

  // Add controlled internal state to properly handle partial inputs
  const [inputValue, setInputValue] = React.useState<string | number>(value ?? '');

  // Sync internal state with prop value
  React.useEffect(() => {
    setInputValue(value ?? '');
  }, [value]);

  return (
    <div className={classNames(props.className)}>
      <FieldLabel label={label} doc={doc} />
      <input
        type="number"
        value={inputValue}
        onChange={e => {
          const rawValue = e.target.value;

          // Update the input display with the raw value
          setInputValue(rawValue);

          // Handle empty or partial inputs
          if (rawValue === '' || rawValue === '-') {
            // For empty or partial negative input, don't call onChange yet
            if (rawValue === '' && allowEmpty) {
              onChange(null);
            }
            return;
          }

          const numValue = Number(rawValue);

          // don't clamp to min/max while typing, it mangles partial input (typing 1024 with
          // min 64 becomes 64024). Clamping happens on blur.
          if (!isNaN(numValue)) {
            onChange(numValue);
          }
        }}
        onBlur={() => {
          const numValue = Number(inputValue);
          if (inputValue === '' || isNaN(numValue)) {
            return;
          }
          let constrainedValue = numValue;
          if (min !== undefined && constrainedValue < min) {
            constrainedValue = min;
          }
          if (max !== undefined && constrainedValue > max) {
            constrainedValue = max;
          }
          if (constrainedValue !== numValue) {
            setInputValue(constrainedValue);
            onChange(constrainedValue);
          }
        }}
        className={inputClasses}
        placeholder={placeholder}
        required={required}
        min={min}
        max={max}
        step="any"
      />
    </div>
  );
};

interface SelectInputPropsBase extends InputProps {
  disabled?: boolean;
  options: GroupedSelectOption[] | SelectOption[];
}

export interface SingleSelectInputProps extends SelectInputPropsBase {
  multiple?: false;
  value: string;
  onChange: (value: string) => void;
}

export interface MultiSelectInputProps extends SelectInputPropsBase {
  multiple: true;
  value: string[];
  onChange: (value: string[]) => void;
}

export type SelectInputProps = SingleSelectInputProps | MultiSelectInputProps;

export const SelectInput = (props: SelectInputProps) => {
  const { label, value, onChange, options, docKey = null, multiple } = props;
  const doc = resolveDoc(props.doc, docKey);

  const flatOptions: SelectOption[] =
    options && options.length > 0
      ? 'options' in options[0]
        ? (options as GroupedSelectOption[]).flatMap(group => group.options)
        : (options as SelectOption[])
      : [];

  const selectedOption = multiple
    ? flatOptions.filter(opt => (value as string[]).includes(opt.value))
    : flatOptions.find(opt => opt.value === (value as string));

  return (
    <div
      className={classNames(props.className, {
        'opacity-30 cursor-not-allowed': props.disabled,
      })}
    >
      <FieldLabel label={label} doc={doc} />
      <Select
        value={selectedOption}
        options={options}
        isDisabled={props.disabled}
        isMulti={multiple}
        className="aitk-react-select-container"
        classNamePrefix="aitk-react-select"
        menuPosition="fixed"
        menuPlacement="auto"
        onChange={selected => {
          if (multiple) {
            const arr = (selected as { value: string }[] | null) ?? [];
            (onChange as (v: string[]) => void)(arr.map(o => o.value));
          } else if (selected) {
            (onChange as (v: string) => void)((selected as { value: string }).value);
          }
        }}
      />
    </div>
  );
};

export interface CreatableSelectInputProps extends InputProps {
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
  options: GroupedSelectOption[] | SelectOption[];
}

const CUSTOM_SELECT_VALUE = '__custom__';

export const CreatableSelectInput = (props: CreatableSelectInputProps) => {
  const { label, value, onChange, options, docKey = null } = props;
  const doc = resolveDoc(props.doc, docKey);

  // Check if current value matches any predefined option
  let isInOptions = false;
  if (options && options.length > 0) {
    if ('options' in options[0]) {
      isInOptions = (options as GroupedSelectOption[]).flatMap(g => g.options).some(opt => opt.value === value);
    } else {
      isInOptions = (options as SelectOption[]).some(opt => opt.value === value);
    }
  }

  const [isCustom, setIsCustom] = React.useState(!isInOptions && !!value);
  const customInputRef = React.useRef<HTMLInputElement>(null);

  // Build select options with "Custom" at the top
  const customOption: SelectOption = { value: CUSTOM_SELECT_VALUE, label: 'Custom' };
  const selectOptions = React.useMemo(() => {
    if (options && options.length > 0 && 'options' in options[0]) {
      return [{ label: '', options: [customOption] }, ...(options as GroupedSelectOption[])];
    }
    return [customOption, ...(options as SelectOption[])];
  }, [options]);

  const selectedOption = isCustom
    ? customOption
    : (() => {
        if (!options || options.length === 0) return undefined;
        if ('options' in options[0]) {
          return (options as GroupedSelectOption[]).flatMap(g => g.options).find(opt => opt.value === value);
        }
        return (options as SelectOption[]).find(opt => opt.value === value);
      })();

  return (
    <div
      className={classNames(props.className, {
        'opacity-30 cursor-not-allowed': props.disabled,
      })}
    >
      <FieldLabel label={label} doc={doc} />
      <div className="flex gap-2">
        <div className={isCustom ? 'w-20 shrink-0' : 'w-full'}>
          <Select
            value={selectedOption}
            options={selectOptions}
            isDisabled={props.disabled}
            className="aitk-react-select-container"
            classNamePrefix="aitk-react-select"
            menuPosition="fixed"
            menuPlacement="auto"
            formatOptionLabel={(option: unknown) => {
              const opt = option as SelectOption;
              return opt.value === CUSTOM_SELECT_VALUE ? (
                <span className="opacity-50 italic">~ Custom ~</span>
              ) : (
                opt.label
              );
            }}
            onChange={selected => {
              if (selected) {
                const val = (selected as { value: string }).value;
                if (val === CUSTOM_SELECT_VALUE) {
                  setIsCustom(true);
                  onChange('');
                  // focus the custom input only when the user actively selects "Custom"
                  requestAnimationFrame(() => customInputRef.current?.focus());
                } else {
                  setIsCustom(false);
                  onChange(val);
                }
              }
            }}
          />
        </div>
        {isCustom && (
          <input
            ref={customInputRef}
            type="text"
            value={value}
            onChange={e => onChange(e.target.value)}
            className={`${inputClasses} flex-1 min-w-0`}
            placeholder={props.placeholder ?? 'Enter custom value'}
            disabled={props.disabled}
          />
        )}
      </div>
    </div>
  );
};

export interface CheckboxProps {
  label?: string | React.ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  className?: string;
  required?: boolean;
  disabled?: boolean;
  docKey?: string | null;
  doc?: ConfigDoc | null;
}

export const Checkbox = (props: CheckboxProps) => {
  const { label, checked, onChange, required, disabled } = props;
  const doc = resolveDoc(props.doc, props.docKey);

  const id = React.useId();

  return (
    <div className={classNames('flex items-center gap-3', props.className)}>
      <button
        type="button"
        role="switch"
        id={id}
        aria-checked={checked}
        aria-required={required}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={classNames(
          'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2',
          checked ? 'bg-blue-500' : 'bg-gray-600',
          disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-opacity-80',
        )}
      >
        <span className="sr-only">Toggle {label}</span>
        <span
          className={classNames(
            'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
            checked ? 'translate-x-5' : 'translate-x-0',
          )}
        />
      </button>
      {label && (
        <>
          <label
            htmlFor={id}
            className={classNames(
              'text-sm font-medium cursor-pointer select-none',
              disabled ? 'text-gray-500' : 'text-gray-300',
            )}
          >
            {doc?.summary ? (
              <HoverTip text={doc.summary}>
                <span className="cursor-help border-b border-dotted border-gray-600">{label}</span>
              </HoverTip>
            ) : (
              label
            )}
          </label>
          {doc && (
            <button
              type="button"
              className="text-gray-500 hover:text-gray-300"
              onClick={() => openDoc(doc)}
              aria-label="More about this setting"
            >
              <CircleHelp className="inline-block h-4 w-4" />
            </button>
          )}
        </>
      )}
    </div>
  );
};

interface FormGroupProps {
  label?: string;
  className?: string;
  docKey?: string | null;
  doc?: ConfigDoc | null;
  children: React.ReactNode;
}

export const FormGroup: React.FC<FormGroupProps> = props => {
  const { label, className, children, docKey = null } = props;
  const doc = resolveDoc(props.doc, docKey);
  return (
    <div className={classNames(className)}>
      <FieldLabel label={label} doc={doc} className="mb-2" />
      <div className="space-y-2">{children}</div>
    </div>
  );
};

export interface SliderInputProps extends InputProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  showValue?: boolean;
}

export const SliderInput: React.FC<SliderInputProps> = props => {
  const { label, value, onChange, min, max, step = 1, disabled, className, docKey = null, showValue = true } = props;
  const doc = resolveDoc(props.doc, docKey);

  const trackRef = React.useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = React.useState(false);

  const clamp = (v: number) => (v < min ? min : v > max ? max : v);
  const snapToStep = (v: number) => {
    if (!Number.isFinite(v)) return min;
    const steps = Math.round((v - min) / step);
    const snapped = min + steps * step;
    return clamp(Number(snapped.toFixed(6)));
  };

  const percent = React.useMemo(() => {
    if (max === min) return 0;
    const p = ((value - min) / (max - min)) * 100;
    return p < 0 ? 0 : p > 100 ? 100 : p;
  }, [value, min, max]);

  const calcFromClientX = React.useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el || !Number.isFinite(clientX)) return;
      const rect = el.getBoundingClientRect();
      const width = rect.right - rect.left;
      if (!(width > 0)) return;

      // Clamp ratio to [0, 1] so it can never flip ends.
      const ratioRaw = (clientX - rect.left) / width;
      const ratio = ratioRaw <= 0 ? 0 : ratioRaw >= 1 ? 1 : ratioRaw;

      const raw = min + ratio * (max - min);
      onChange(snapToStep(raw));
    },
    [min, max, step, onChange],
  );

  // Mouse/touch pointer drag
  const onPointerDown = (e: React.PointerEvent) => {
    if (disabled) return;
    e.preventDefault();

    // Capture the pointer so moves outside the element are still tracked correctly
    try {
      (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    } catch {}

    setDragging(true);
    calcFromClientX(e.clientX);

    const handleMove = (ev: PointerEvent) => {
      ev.preventDefault();
      calcFromClientX(ev.clientX);
    };
    const handleUp = (ev: PointerEvent) => {
      setDragging(false);
      // release capture if we got it
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
      } catch {}
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
  };

  return (
    <div className={classNames(className, disabled ? 'opacity-30 cursor-not-allowed' : '')}>
      <FieldLabel label={label} doc={doc} />

      <div className="flex items-center gap-3">
        <div className="flex-1">
          <div
            ref={trackRef}
            onPointerDown={onPointerDown}
            className={classNames(
              'relative w-full h-6 select-none outline-none',
              disabled ? 'pointer-events-none' : 'cursor-pointer',
            )}
          >
            {/* Thicker track */}
            <div className="pointer-events-none absolute left-0 right-0 top-1/2 -translate-y-1/2 h-3 rounded-sm bg-gray-800 border border-gray-700" />

            {/* Fill */}
            <div
              className="pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 h-3 rounded-sm bg-blue-500"
              style={{ width: `${percent}%` }}
            />

            {/* Thumb */}
            <div
              onPointerDown={onPointerDown}
              className={classNames(
                'absolute top-1/2 -translate-y-1/2 -ml-2',
                'h-4 w-4 rounded-full bg-white shadow border border-gray-300 cursor-pointer',
                'after:content-[""] after:absolute after:inset-[-6px] after:rounded-full after:bg-transparent', // expands hit area
                dragging ? 'ring-2 ring-blue-600' : '',
              )}
              style={{ left: `calc(${percent}% )` }}
            />
          </div>

          <div className="flex justify-between text-xs text-gray-500 mt-0.5 select-none">
            <span>{min}</span>
            <span>{max}</span>
          </div>
        </div>

        {showValue && (
          <div className="min-w-[3.5rem] text-right text-sm px-3 py-1 bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm">
            {Number.isFinite(value) ? value : ''}
          </div>
        )}
      </div>
    </div>
  );
};
