const assert = require('node:assert/strict');
const test = require('node:test');
const sqlite3 = require('sqlite3');

const { migrateNoisingConfig } = require('../src/helpers/noisingConfig');

const run = (db: any, sql: string, params: unknown[] = []) =>
  new Promise<void>((resolve, reject) => {
    db.run(sql, params, (error: Error | null) => (error ? reject(error) : resolve()));
  });

const get = (db: any, sql: string) =>
  new Promise<any>((resolve, reject) => {
    db.get(sql, (error: Error | null, row: unknown) => (error ? reject(error) : resolve(row)));
  });

const closeDatabase = (db: any) =>
  new Promise<void>((resolve, reject) => {
    db.close((error: Error | null) => (error ? reject(error) : resolve()));
  });

test('legacy train config receives complete disabled noising defaults without adding loss_split', () => {
  const train: any = { optimizer: 'adamw8bit' };

  migrateNoisingConfig(train);

  assert.deepEqual(train.weight_noise, {
    enabled: false,
    mode: 'relative',
    sigma: 0.00125,
    bound_norm: false,
    log_every: 50,
  });
  assert.deepEqual(train.gradient_noise, {
    enabled: false,
    mode: 'neelakantan',
    sigma: 0.001,
    eta: 0.01,
    gamma: 0.55,
    log_every: 50,
  });
  assert.equal('loss_split' in train, false);
});

test('partial saved noising values override defaults while omitted fields migrate', () => {
  const train: any = {
    weight_noise: { enabled: true, sigma: 0.02 },
    gradient_noise: { enabled: true, mode: 'absolute', sigma: 0.03 },
  };

  migrateNoisingConfig(train);

  assert.deepEqual(train.weight_noise, {
    enabled: true,
    mode: 'relative',
    sigma: 0.02,
    bound_norm: false,
    log_every: 50,
  });
  assert.deepEqual(train.gradient_noise, {
    enabled: true,
    mode: 'absolute',
    sigma: 0.03,
    eta: 0.01,
    gamma: 0.55,
    log_every: 50,
  });
});

test('all noising controls survive a SQLite JSON save and reload', async () => {
  const db = new sqlite3.Database(':memory:');
  try {
    await run(db, 'CREATE TABLE jobs (job_config TEXT NOT NULL)');
    const jobConfig = {
      config: {
        process: [
          {
            train: {
              weight_noise: {
                enabled: true,
                mode: 'absolute',
                sigma: 0.012,
                bound_norm: true,
                log_every: 17,
              },
              gradient_noise: {
                enabled: true,
                mode: 'relative',
                sigma: 0.034,
                eta: 0.056,
                gamma: 0.78,
                log_every: 19,
              },
            },
          },
        ],
      },
    };
    await run(db, 'INSERT INTO jobs (job_config) VALUES (?)', [JSON.stringify(jobConfig)]);

    const row = await get(db, 'SELECT job_config FROM jobs LIMIT 1');
    const reloaded = JSON.parse(row.job_config);
    migrateNoisingConfig(reloaded.config.process[0].train);

    assert.deepEqual(reloaded, jobConfig);
  } finally {
    await closeDatabase(db);
  }
});
