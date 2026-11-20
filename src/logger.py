import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Projektroten: den här filen ligger i src/, så rotkatalogen är ett steg
# upp. Sökvägen till loggkatalogen måste vara absolut — den tidigare
# relativa sökvägen ("logs/project.log") skapade en loggkatalog i vilken
# arbetskatalog som helst (t.ex. en spillkatalog notebooks/logs/).
PROJEKT_ROT = Path(__file__).resolve().parent.parent

# Delad filhandler: alla loggrar pekar på SAMMA handler-instans, så att
# rotationen (omdöpning av filen vid midnatt) bara görs en gång. Får
# i stället varje logger en egen handler mot samma fil misslyckas
# omdöpet på Windows, eftersom filen då hålls öppen av flera referenser.
_fil_handler = None


def setup_logger(name: str = "pm25") -> logging.Logger:

    logg_dir = PROJEKT_ROT / "logs"
    logg_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Rotation vid varje midnatt: project.log skrivs om dagligen och de
    # 14 senaste dagliga backuperna (project.log.2026-11-14 osv.)
    # behålls. Utan rotation växer project.log i all evighet.
    global _fil_handler
    if _fil_handler is None:
        _fil_handler = TimedRotatingFileHandler(
            logg_dir / "project.log",
            when="midnight",
            backupCount=14,
            encoding="utf-8",
        )
    _fil_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(_fil_handler)

    return logger




#Test-sektion för logger

if __name__ == "__main__":
    logger = setup_logger()

    logger.debug("Detta är en debug-logg")
    logger.info("Detta är en info-logg")
    logger.warning("Detta är en warning-logg")
    logger.error("Detta är en error-logg")
