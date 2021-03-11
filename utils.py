# cofre libraries

from datetime import date
from datetime import datetime
from typing import Tuple, Dict

# third-party libraries
import numpy as np
import pandas as pd
import pymc3 as pm
import torch
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

# our code base
from model import MCMCModel

state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Puerto Rico": "PR",
}


def _clean_usafacts_county_names(usafacts_df: pd.DataFrame) -> pd.DataFrame:
    """Internal: apply some county-name cleaning functinos to any USA Facts dataset"""

    # For Virginia: remove "City" or "city" at the *end* of county names (that's the $)
    usafacts_df.loc[(usafacts_df.state == "VA"), "county"] = (
        usafacts_df[usafacts_df.state == "VA"]
            .county.str.replace(r" City$", "")
            .str.replace(r" city$", "")
            .str.strip()
    )

    # These are spelled wrong
    usafacts_df.loc[
        (usafacts_df.state == "VA") & (usafacts_df.county == "Matthews County"),
        "county",
    ] = "Mathews County"

    usafacts_df.loc[
        (usafacts_df.state == "NM") & (usafacts_df.county == "DoÔøΩa Ana County"),
        "county",
    ] = "Dona Ana County"

    # this needs capitalization
    usafacts_df.loc[
        (usafacts_df["state"] == "MN")
        & (usafacts_df["county"] == "Lac qui Parle County"),
        "county",
    ] = "Lac Qui Parle County"

    # For Missouri: replace "Jackson County (including other portions of Kansas City)"
    long_name = "Jackson County (including other portions of Kansas City)"
    usafacts_df.loc[
        (usafacts_df.state == "MO") & (usafacts_df.county == long_name), "county",
    ] = "Jackson County"

    # this needs the "and City" suffix removed
    usafacts_df.loc[
        (usafacts_df["state"] == "CO")
        & (usafacts_df["county"] == "Broomfield County and City"),
        "county",
    ] = "Broomfield County"

    return usafacts_df


def _clean_google_data(raw_google_data: pd.DataFrame) -> pd.DataFrame:
    """Given raw Google mobility data, clean it and redownload it."""

    df = raw_google_data[
        raw_google_data["country_region"] == "United States"
        ].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["state"] = df["sub_region_1"].map(state_abbreviations)
    df = (
        df.drop(["country_region_code", "country_region"], axis="columns")
            .rename(
            {"sub_region_1": "state_name", "sub_region_2": "county"}, axis="columns"
        )
            .fillna({"county": "statewide", "state": "all US", "state_name": "all US"})
    )

    # this has an ñ that doesn't match up
    df.loc[
        (df["state"] == "NM") & (df["county"] == "Doña Ana County"), "county"
    ] = "Dona Ana County"

    # this was renamed in 2015
    df.loc[
        (df["state"] == "SD") & (df["county"] == "Shannon County"), "county"
    ] = "Oglala Lakota County"

    # these need the County suffix
    df.loc[
        (df["state"] == "MO") & (df["county"] == "St. Louis"), "county"
    ] = "St. Louis County"
    df.loc[
        (df["state"] == "MD") & (df["county"] == "Baltimore"), "county"
    ] = "Baltimore County"

    # this needs capitalization
    df.loc[
        (df["state"] == "MN") & (df["county"] == "Lac qui Parle County"), "county"
    ] = "Lac Qui Parle County"

    # merge to county
    cases = load_county_case_data()
    merged = df.merge(cases, on=["state", "county", "date"], how="left", indicator=True)
    merged = merged.drop(["cases", "new_cases", "_merge", "stateFIPS"], axis=1)

    merged.to_csv("data/google_mobility_clean.csv", index=False)


def load_county_case_data() -> pd.DataFrame:
    """
    Read and clean county-level COVID case data.
    """

    df = pd.read_csv("data/covid_confirmed_usafacts.csv")
    df = pd.melt(
        df,
        id_vars=["countyFIPS", "County Name", "State", "stateFIPS"],
        var_name="date",
        value_name="cases",
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename({"County Name": "county", "State": "state"}, axis="columns")

    # Add "new_cases" column
    df["cases_previous"] = (
        df.groupby(["countyFIPS", "county", "state", "stateFIPS"])["cases"]
            .shift(1)
            .fillna(0)
    )
    df["cases"] = df["cases"] - df["cases_previous"]
    df = df.drop("cases_previous", axis=1)
    df = _clean_usafacts_county_names(df)

    return df


def read_deaths_data() -> pd.DataFrame:
    """
    Read and clean county-level COVID deaths data.
    """

    df = pd.read_csv("data/covid_deaths_usafacts.csv")
    df = pd.melt(
        df,
        id_vars=["countyFIPS", "County Name", "State", "stateFIPS"],
        var_name="date",
        value_name="deaths",
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename({"County Name": "county", "State": "state"}, axis="columns")

    # Add "new_deaths" column
    df["deaths_previous"] = (
        df.groupby(["countyFIPS", "county", "state", "stateFIPS"])["deaths"]
            .shift(1)
            .fillna(0)
    )
    df["new_deaths"] = df["deaths"] - df["deaths_previous"]
    df = df.drop("deaths_previous", axis=1)
    df = _clean_usafacts_county_names(df)

    return df


def read_google_mobility_data() -> pd.DataFrame:
    """
    Read United States county-level mobility data from Google.
    """

    df = pd.read_csv("data/google_mobility_clean.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def read_geo_data() -> pd.DataFrame:
    """Read geo mapping"""

    df = pd.read_parquet("data/geo_map")
    return df


def read_kinsa_data() -> pd.DataFrame:
    """Read Kinsa smart thermometer data"""

    df = pd.read_csv("data/kinsa.csv", index_col=0)
    return df


def compute_mean_kinsa_atypical_illness(df: pd.DataFrame, state: str) -> pd.DataFrame:
    state_data = df[df["state"] == state]
    state_data.drop(
        ['region_id', 'region_name', 'state', 'doy', 'atypical_ili_delta', 'anomaly_fevers', 'forecast_expected',
         'forecast_lower', 'forecast_upper'], inplace=True, axis=1)
    mean_data = state_data.groupby(["date"]).agg(['mean'])
    mean_data = mean_data.stack().reset_index().drop(["level_1"], axis=1)
    mean_data.reset_index(inplace=True)
    mean_data["date"] = pd.to_datetime(mean_data.date, format='%Y-%m-%d', errors='coerce')
    mean_data.set_index("date", inplace=True)
    mean_data.drop(["index"], inplace=True, axis=1)
    return mean_data


def merge_mobility_and_cases(
        mobility: pd.DataFrame, cases: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge mobility data and case data.

    We do a left join from mobility to cases, and verify that the only rows in the left
    that don't get matched are either "statewide" rows (mobility changes for entire
    states) or counties in Alaska, which we can just drop.
    """

    merged = mobility.merge(
        cases, on=["state", "county", "date"], how="left", indicator=True
    )

    rows_without_match = (
        merged[merged["_merge"] == "left_only"]
            .groupby(["state_name", "county"])
            .size()
            .reset_index(name="size")
    )

    # Check that every row is either (1) Alaska or (2) statewide
    criteria = (rows_without_match.state_name == "Alaska") | (
            (rows_without_match.state_name != "Alaska")
            & (rows_without_match.county == "statewide")
    )
    assert len(rows_without_match[criteria]) == len(
        rows_without_match
    ), "some rows didn't get matched"

    return merged


def confirmed_to_onset(confirmed: np.array, p_delay: np.array) -> pd.Series:
    assert not confirmed.isna().any()

    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1], periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)

    return onset


def adjust_onset_for_right_censorship(onset: pd.Series, p_delay: np.array) -> Tuple[np.array, np.array]:
    cumulative_p_delay = p_delay.cumsum()

    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)

    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)

    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay

    return adjusted, cumulative_p_delay


def load_daily_state_cases(url: str = "https://covidtracking.com/api/v1/states/daily.csv") \
        -> pd.DataFrame:
    states = pd.read_csv(url,
                         parse_dates=['date'],
                         index_col=['state', 'date']).sort_index()
    # Note: GU/AS/VI do not have enough data for this model to run
    # Note: PR had -384 change recently in total count so unable to model
    states = states.drop(['MP', 'GU', 'AS', 'PR', 'VI'])

    # Errors in Covidtracking.com
    states.loc[('WA', '2020-04-21'), 'positive'] = 12512
    states.loc[('WA', '2020-04-22'), 'positive'] = 12753
    states.loc[('WA', '2020-04-23'), 'positive'] = 12753 + 190

    states.loc[('VA', '2020-04-22'), 'positive'] = 10266
    states.loc[('VA', '2020-04-23'), 'positive'] = 10988

    states.loc[('PA', '2020-04-22'), 'positive'] = 35684
    states.loc[('PA', '2020-04-23'), 'positive'] = 37053

    states.loc[('MA', '2020-04-20'), 'positive'] = 39643

    states.loc[('CT', '2020-04-18'), 'positive'] = 17550
    states.loc[('CT', '2020-04-19'), 'positive'] = 17962

    states.loc[('HI', '2020-04-22'), 'positive'] = 586

    states.loc[('RI', '2020-03-07'), 'positive'] = 3

    # Make sure that all the states have current data
    today = datetime.combine(date.today(), datetime.min.time())
    last_updated = states.reset_index('date').groupby('state')['date'].max()
    is_current = last_updated < today

    try:
        assert is_current.sum() == 0
    except AssertionError:
        print("Not all states have updated")
        print(last_updated[is_current])

    # Ensure all case diffs are greater than zero
    for state, grp in states.groupby('state'):
        new_cases = grp.positive.diff().dropna()
        is_positive = new_cases.ge(0)

        try:
            assert is_positive.all()
        except AssertionError:
            print(f"Warning: {state} has date with negative case counts")
            print(new_cases[~is_positive])

    # Let's make sure that states have added cases
    idx = pd.IndexSlice
    assert not states.loc[idx[:, '2020-04-22':'2020-04-23'], 'positive'].groupby('state').diff().dropna().eq(0).any()
    # states["positive"] =  states.positive.diff().dropna()
    # states.rename(columns={"positive": "confirmed"}, inplace=True)
    return states


def load_patient_data():
    # Load the patient CSV
    patients = pd.read_csv(
        "data/linelist.csv",
        parse_dates=False,
        usecols=[
            "date_confirmation",
            "date_onset_symptoms"],
        low_memory=False)

    patients.columns = ["Onset", "Confirmed"]

    # There's an errant reversed date
    patients = patients.replace("01.31.2020", "31.01.2020")

    # Must have strings that look like individual dates
    # "2020.03.09" is 10 chars long
    is_ten_char = lambda x: x.str.len().eq(10)
    patients = patients[is_ten_char(patients.Confirmed) &
                        is_ten_char(patients.Onset)]

    # Convert both to datetimes
    patients.Confirmed = pd.to_datetime(patients.Confirmed, format='%d.%m.%Y')

    patients['Timestamp'] = pd.to_datetime(patients.Onset, format='%d-%m-%Y', errors='coerce')
    mask = patients.Timestamp.isnull()
    patients.loc[mask, 'Timestamp'] = pd.to_datetime(patients[mask]['Onset'], format='%d.%m.%Y', errors='coerce')
    patients.drop(["Onset"], axis=1, inplace=True)
    patients.rename(columns={"Timestamp": "Onset"}, inplace=True)
    patients.dropna(inplace=True)

    # Only keep records where confirmed > onset
    patients = patients[patients.Confirmed >= patients.Onset]
    return patients


def load_rt_predictions(path="./data/projections/2020-05-23/US_NY.csv"):
    ny_rt = pd.read_csv(path)
    r_values_mean = ny_rt["r_values_mean"]
    r_values_mean.dropna(inplace=True)
    X = torch.arange(0, r_values_mean.size, 1).double()
    y = torch.from_numpy(r_values_mean.values)
    return X, y


def get_delay_onset_confirmation_probabilities(patients: pd.DataFrame) -> np.array:
    # Calculate the delta in days between onset and confirmation
    delay = (patients.Confirmed - patients.Onset).dt.days

    # Convert samples to an empirical distribution
    p_delay = delay.value_counts().sort_index()
    new_range = np.arange(0, p_delay.index.max() + 1)
    p_delay = p_delay.reindex(new_range, fill_value=0)
    p_delay /= p_delay.sum()
    return p_delay


def confirmed_to_onset(confirmed: pd.DataFrame, p_delay: np.array) -> pd.Series:
    assert not confirmed.isna().any()

    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                       periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)

    return onset


def adjust_onset_for_right_censorship(onset: pd.Series, p_delay: np.array) -> Tuple[pd.Series, np.array]:
    cumulative_p_delay = p_delay.cumsum()

    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)

    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)

    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay

    return adjusted, cumulative_p_delay


def compute_adjusted_cases(state: str, df: pd.DataFrame, p_delay) -> Tuple[pd.Series, np.array]:
    original, smoothed = prepare_cases(df, state)
    smoothed = smoothed.loc["2020-04-01":]
    onset = confirmed_to_onset(smoothed, p_delay)
    onset = onset.loc["2020-04-01":]
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return adjusted, cumulative_p_delay


def compute_delay_onset_confirmation_probabilities() -> np.array:
    patients = load_patient_data()
    patients = patients[(patients["Onset"] >= "2020-04-01")]
    return get_delay_onset_confirmation_probabilities(patients)


def agg_cases_to_state_level(county_cases) -> pd.DataFrame:
    """Aggregate county cases to state level"""

    state_cases = county_cases.groupby(["state", "date"]).agg(new_cases="sum", cases="sum")

    # Add "new_cases_7day_avg" column
    state_cases["new_cases_7day_avg"] = (
        state_cases.sort_values(["state", "date"])
            .new_cases.groupby("state")
            .rolling(7, min_periods=1, win_type="gaussian", center=False)
            .mean(std=2)
            .round()
            .reset_index(0, drop=True)
    )

    return state_cases


def prepare_county_cases(cases_df: pd.DataFrame, cutoff=25):
    new_cases = cases_df.cases

    smoothed = new_cases.rolling(7,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()

    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


def prepare_cases(states: pd.DataFrame, state: str, cutoff=25):
    new_cases = states.xs(state).positive.diff().dropna()

    smoothed = new_cases.rolling(7,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()

    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


def df_from_model(model: MCMCModel) -> pd.DataFrame:
    r_t = model.trace['r_t']
    mean = np.mean(r_t, axis=0)
    median = np.median(r_t, axis=0)
    hpd_90 = pm.stats.hpd(r_t, credible_interval=.9)
    hpd_50 = pm.stats.hpd(r_t, credible_interval=.5)

    idx = pd.MultiIndex.from_product([
        [model.region],
        model.trace_index
    ], names=['region', 'date'])

    df = pd.DataFrame(data=np.c_[mean, median, hpd_90, hpd_50], index=idx,
                      columns=['mean', 'median', 'lower_90', 'upper_90', 'lower_50', 'upper_50'])
    return df


def create_and_run_model(name: str, states: pd.DataFrame, p_delay: np.array) -> MCMCModel:
    original, smoothed = prepare_cases(states, name)
    smoothed = smoothed.loc["2020-04-01":]
    onset = confirmed_to_onset(smoothed, p_delay)
    onset = onset.loc["2020-04-01":]
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return MCMCModel(name, onset, cumulative_p_delay).run()


def create_and_run_model_for_counties(name: str, cases_df: pd.DataFrame, p_delay: np.array) -> MCMCModel:
    original, smoothed = prepare_county_cases(cases_df)
    onset = confirmed_to_onset(smoothed, p_delay)
    onset = onset.loc["2020-04-01":]
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return MCMCModel(name, onset, cumulative_p_delay).run_gp()


def reparametrize(models: Dict[str, MCMCModel]):
    # Check to see if there were divergences
    n_diverging = lambda x: x.trace['diverging'].nonzero()[0].size
    divergences = pd.Series([n_diverging(m) for m in models.values()], index=models.keys())
    has_divergences = divergences.gt(0)

    print('Diverging states:')
    print(divergences[has_divergences])

    # Rerun states with divergences
    for state, n_divergences in divergences[has_divergences].items():
        models[state].run()


def aggregate_results(models: Dict[str, MCMCModel]) -> pd.DataFrame:
    results = None

    for state, model in models.items():

        df = df_from_model(model)

        if results is None:
            results = df
        else:
            results = pd.concat([results, df], axis=0)
    return results


def plot_rt(name: str, result: pd.DataFrame, ax, c=(.3, .3, .3, 1), ci=(0, 0, 0, .05)):
    ax.set_ylim(0, 1.6)
    ax.set_title(name, fontsize=20)
    ax.plot(result['median'],
            marker='o',
            markersize=4,
            markerfacecolor='w',
            lw=1,
            c=c,
            markevery=2)
    ax.fill_between(
        result.index,
        result['lower_90'].values,
        result['upper_90'].values,
        color=ci,
        lw=0)
    ax.axhline(1.0, linestyle=':', lw=1)
    ax.set_ylabel(r"$R_t$", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.gcf().subplots_adjust(left=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=100))
